"""Molecular dynamics integrators.

Every integrator is a small stateless object holding its parameters, with

.. code-block:: python

    state = integrator.step(state, potential)

as the only method that matters.  The extended-system variables (Nose-Hoover
chains, barostat strain rate) live on the :class:`~atomlab.md.state.MDState`,
not on the integrator, so a restart is a single picklable object and two runs
can share one integrator instance without cross-talk.

.. note::
   ``step`` mutates ``state`` in place and returns it.  Copying a few hundred
   thousand floats per step to fake immutability would dominate the cost of a
   cheap potential.  Use :meth:`MDState.copy` if you need to keep a snapshot.

What is here and why
--------------------

:class:`VelocityVerlet`
    NVE.  Symplectic, time-reversible, second order.  It is the reference
    against which everything else is judged, because for it the *exact*
    statement is available: it conserves a shadow Hamiltonian ``H + O(dt^2)``
    to all times, so the total energy oscillates with an amplitude ``~dt^2``
    but does not drift secularly.

:class:`Langevin`
    NVT by stochastic thermostatting, integrated with the **BAOAB** splitting.

:class:`NoseHooverChain`
    NVT by deterministic extended dynamics, with Suzuki-Yoshida decomposition
    of the thermostat propagator and a conserved quantity that makes the
    implementation falsifiable.

:class:`MTKBarostat`
    Isotropic NPT (Martyna-Tobias-Klein), also with an exposed conserved
    quantity.

:class:`BerendsenThermostat`
    A negative control.  It does **not** sample the canonical ensemble.  See
    its docstring; it is included precisely because it looks fine by the usual
    diagnostic (the mean temperature is right) while the physics is wrong,
    which is the thesis of this repository in miniature.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from ..units import BAR_TO_EV_A3, KB, MVV2E
from .state import MDState
from .velocities import remove_com_momentum

__all__ = [
    "Integrator",
    "VelocityVerlet",
    "Langevin",
    "NoseHooverChain",
    "MTKBarostat",
    "BerendsenThermostat",
    "suzuki_yoshida_weights",
]


# --------------------------------------------------------------------------
# small numerical helpers
# --------------------------------------------------------------------------


def _accelerations(forces: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """``a = F / (m * MVV2E)`` in A/ps^2, from forces in eV/A and masses in amu."""
    return forces / (masses[:, None] * MVV2E)


def _sinhx_over_x(x: float) -> float:
    """``sinh(x)/x``, with the removable singularity handled by its Taylor series.

    Used by the MTK propagator, where ``x`` is a strain increment of order
    ``1e-6``.  The naive quotient loses all significance there, and the error
    lands directly in the positions.
    """
    if abs(x) < 1e-6:
        x2 = x * x
        return 1.0 + x2 / 6.0 * (1.0 + x2 / 20.0)
    return math.sinh(x) / x


def suzuki_yoshida_weights(order: int) -> np.ndarray:
    """Symmetric Suzuki-Yoshida weights of the requested order.

    Parameters
    ----------
    order : {1, 3, 5, 7}
        Number of sub-steps.  The weights sum to one and are palindromic, so
        composing a second-order symmetric propagator with them raises the
        order to ``order + 1`` (i.e. 4th order for ``order = 3``).

    Returns
    -------
    ndarray, shape (order,)

    Notes
    -----
    The higher orders necessarily contain a **negative** weight -- a backwards
    sub-step -- because no composition of positive time steps can exceed second
    order (Suzuki's theorem).  That is harmless for the thermostat propagator,
    which is analytically integrable in each factor, and is why the technique is
    applied to the thermostat rather than to the physical forces.
    """
    if order == 1:
        return np.array([1.0])
    if order == 3:
        w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        return np.array([w1, 1.0 - 2.0 * w1, w1])
    if order == 5:
        w1 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        return np.array([w1, w1, 1.0 - 4.0 * w1, w1, w1])
    if order == 7:
        w1 = 0.784513610477560
        w2 = 0.235573213359357
        w3 = -1.177679984178870
        w4 = 1.0 - 2.0 * (w1 + w2 + w3)
        return np.array([w1, w2, w3, w4, w3, w2, w1])
    raise ValueError(f"Suzuki-Yoshida order must be one of 1, 3, 5, 7; got {order}")


def _nhc_propagate(
    two_ke: float,
    n_dof_target: int,
    kt: float,
    xi: np.ndarray,
    v_xi: np.ndarray,
    q: np.ndarray,
    dt_therm: float,
    weights: np.ndarray,
    n_respa: int,
) -> float:
    """Propagate one Nose-Hoover chain and return the velocity scale factor.

    This is the Martyna-Tuckerman-Tobias-Klein chain propagator: a symmetric
    (hence time-reversible) composition in which each chain velocity is updated
    inside the exponential of the *next* one, so the coupling term
    ``-v_xi[k+1] v_xi[k]`` is integrated exactly rather than by a Taylor
    expansion.  A naive simultaneous update of the chain is unstable for stiff
    chains and destroys the conserved quantity, which is the whole reason for
    this rather baroque ordering.

    Parameters
    ----------
    two_ke : float
        ``sum m v^2`` in eV for the thermostatted subsystem, i.e. twice its
        kinetic energy.
    n_dof_target : int
        Number of degrees of freedom the chain is driving to ``kt`` each.  The
        first chain element couples to ``two_ke - n_dof_target * kt``.
    kt : float
        ``k_B T`` in eV.
    xi, v_xi : ndarray, shape (M,)
        Chain positions (dimensionless) and velocities (1/ps).  Updated in place.
    q : ndarray, shape (M,)
        Chain masses in eV ps^2.
    dt_therm : float
        Time interval to propagate the chain over, in ps.  In a Trotter
        factorisation this is half the MD timestep, applied at each end.
    weights : ndarray, shape (n_sy,)
        Suzuki-Yoshida weights.
    n_respa : int
        Number of equal sub-divisions of ``dt_therm`` (multiple time stepping);
        raise it when ``tau`` is much shorter than the MD timestep.

    Returns
    -------
    float
        Factor by which the subsystem velocities must be multiplied.  The
        caller applies it, because for the particle chain that means scaling
        an ``(N, 3)`` array and for the barostat chain a single scalar.
    """
    m = xi.size
    scale = 1.0

    g = np.empty(m)
    g[0] = (two_ke - n_dof_target * kt) / q[0]
    for k in range(1, m):
        g[k] = (q[k - 1] * v_xi[k - 1] ** 2 - kt) / q[k]

    for _ in range(n_respa):
        for w in weights:
            delta = w * dt_therm / n_respa
            d2, d4, d8 = 0.5 * delta, 0.25 * delta, 0.125 * delta

            # backward sweep: outermost chain element first
            v_xi[m - 1] += g[m - 1] * d4
            for k in range(m - 2, -1, -1):
                e = math.exp(-v_xi[k + 1] * d8)
                v_xi[k] = (v_xi[k] * e + g[k] * d4) * e

            # scale the subsystem; two_ke follows so that G[0] stays consistent
            s = math.exp(-v_xi[0] * d2)
            scale *= s
            two_ke *= s * s

            xi += v_xi * d2

            # forward sweep with the updated G[0]
            g[0] = (two_ke - n_dof_target * kt) / q[0]
            for k in range(m - 1):
                e = math.exp(-v_xi[k + 1] * d8)
                v_xi[k] = (v_xi[k] * e + g[k] * d4) * e
                g[k + 1] = (q[k] * v_xi[k] ** 2 - kt) / q[k + 1]
            v_xi[m - 1] += g[m - 1] * d4

    return scale


def _chain_energy(xi: np.ndarray, v_xi: np.ndarray, q: np.ndarray, n_dof_target: int, kt: float) -> float:
    """Energy stored in a Nose-Hoover chain, in eV.

    ``sum_k 0.5 Q_k v_xi_k^2 + n_dof_target * kT * xi_0 + kT * sum_{k>0} xi_k``.
    The asymmetry in the potential term is not a typo: only the first chain
    element thermostats ``n_dof_target`` degrees of freedom, each subsequent one
    thermostats exactly one (the previous chain element).
    """
    kinetic = 0.5 * float(np.sum(q * v_xi**2))
    potential = kt * (n_dof_target * xi[0] + float(np.sum(xi[1:])))
    return kinetic + potential


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------


class Integrator(abc.ABC):
    """Base class: advance an :class:`MDState` by one timestep.

    Attributes
    ----------
    dt : float
        Timestep in ps.  Note the unit: 1 fs is ``dt = 0.001``.
    name : str
        Short identifier used in logs and figures.
    """

    name: str = "integrator"

    def __init__(self, dt: float, *, virial: bool = True) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0 ps, got {dt}")
        self.dt = float(dt)
        #: Whether force evaluations also request the virial.  Barostats force
        #: this on; for NVE/NVT with an expensive model it can be turned off.
        self.virial = bool(virial)

    @abc.abstractmethod
    def step(self, state: MDState, potential) -> MDState:
        """Advance ``state`` by :attr:`dt`, in place, and return it."""

    def run(self, state: MDState, potential, n_steps: int) -> MDState:
        """Take ``n_steps`` steps.  Convenience wrapper; no logging.

        The real driver with logging, striding and restarts is
        :mod:`atomlab.md.simulate`.
        """
        for _ in range(int(n_steps)):
            state = self.step(state, potential)
        return state

    def conserved_quantity(self, state: MDState) -> float:
        """Quantity this integrator conserves exactly in the ``dt -> 0`` limit, in eV.

        Raises
        ------
        NotImplementedError
            For stochastic integrators, which conserve nothing pathwise.
        """
        raise NotImplementedError(f"{type(self).__name__} has no conserved quantity")

    def initialize(self, state: MDState, potential) -> MDState:
        """Prepare ``state`` for this integrator (force evaluation, chain setup)."""
        if state.result is None or (self.virial and state.result.virial is None):
            state.refresh(potential, virial=self.virial)
        return state

    def _half_kick(self, state: MDState, dt: float) -> None:
        state.velocities += dt * _accelerations(state.forces, state.masses)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(dt={self.dt})"


# --------------------------------------------------------------------------
# NVE
# --------------------------------------------------------------------------


class VelocityVerlet(Integrator):
    """Velocity Verlet (NVE).

    .. code-block:: text

        v(t + dt/2) = v(t)        + (dt/2) F(t)/m
        r(t + dt)   = r(t)        + dt v(t + dt/2)
        v(t + dt)   = v(t + dt/2) + (dt/2) F(t + dt)/m

    One force evaluation per step, reusing the end-of-step forces as the
    start-of-step forces.

    Being the composition ``exp(dt/2 L_v) exp(dt L_r) exp(dt/2 L_v)`` of exact
    flows of two Hamiltonians, it is symplectic and exactly time-reversible:
    running ``n`` steps forward, negating all velocities and running ``n`` more
    returns the initial positions to round-off.  That is a far stronger test of
    an implementation than energy conservation, because a wrong-but-consistent
    force sign or mass factor still conserves *something*.

    Parameters
    ----------
    dt : float
        Timestep in ps.
    virial : bool
        Request the virial at each force evaluation (needed only for pressure
        logging).
    """

    name = "velocity-verlet"

    def step(self, state: MDState, potential) -> MDState:
        self.initialize(state, potential)
        dt = self.dt

        self._half_kick(state, 0.5 * dt)
        state.positions = state.positions + dt * state.velocities
        state.refresh(potential, virial=self.virial)
        self._half_kick(state, 0.5 * dt)

        state.step += 1
        state.time += dt
        return state

    def conserved_quantity(self, state: MDState) -> float:
        """Total energy ``KE + U`` in eV."""
        return state.total_energy()


# --------------------------------------------------------------------------
# Langevin / BAOAB
# --------------------------------------------------------------------------


class Langevin(Integrator):
    r"""Langevin dynamics integrated with the BAOAB splitting (NVT).

    The Langevin equation in metal units is

    .. math::
        \dot r = v, \qquad
        m\,\text{MVV2E}\,\dot v = F(r) - \gamma\, m\,\text{MVV2E}\, v
                                  + \sqrt{2 \gamma\, m\,\text{MVV2E}\, k_B T}\, \eta(t)

    with :math:`\gamma` the friction in 1/ps.  Splitting the generator into
    ``A`` (drift, :math:`\dot r = v`), ``B`` (kick, :math:`\dot v = F/m`) and
    ``O`` (Ornstein-Uhlenbeck, the friction plus noise, integrated *exactly*)
    leaves a choice of ordering, and the choice matters far more than one would
    guess.

    **Why BAOAB.**  All symmetric splittings are weakly second order, but their
    *configurational* averages -- which is what every observable in this
    repository is -- have very different error constants.  Leimkuhler and
    Matthews (*AMRX* 2013; *J. Chem. Phys.* **138**, 174102) showed that BAOAB
    has a leading configurational error of order :math:`dt^2` whose coefficient
    vanishes in the high-friction limit, giving effective fourth-order accuracy
    there, whereas OBABO and ABOBA retain an :math:`O(dt^2)` bias with a
    non-vanishing coefficient.  Concretely, for a harmonic oscillator of
    frequency :math:`\omega`, BAOAB samples

    .. math::  \langle x^2 \rangle = \frac{k_B T}{k}\,\frac{1}{1 - \omega^2 dt^2/4}

    independent of the friction -- a 0.25% bias at :math:`\omega\,dt = 0.1` --
    while the Euler-Maruyama scheme that a naive implementation produces is only
    first order and biases the temperature itself.  Since the whole project
    hinges on resolving small differences between potentials, an integrator bias
    that is not small compared with those differences would be fatal.

    The ``O`` factor is the exact solution of the Ornstein-Uhlenbeck process,
    :math:`v \to c_1 v + c_2 \sigma \xi` with :math:`c_1 = e^{-\gamma dt}`,
    :math:`c_2 = \sqrt{1 - c_1^2}` and
    :math:`\sigma_i = \sqrt{k_B T/(m_i \text{MVV2E})}`, so it is stable for any
    friction, however large.

    Parameters
    ----------
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in K.
    friction : float
        Friction coefficient :math:`\gamma` in 1/ps.  For argon, ``1-10 /ps``
        thermalises in a few ps without badly perturbing the dynamics; very
        large values sample well but destroy transport coefficients.
    rng : numpy.random.Generator, optional
        Noise source.  If omitted, ``state.rng`` is used, and it is an error
        for both to be missing -- there is no implicit global seed anywhere in
        this package.
    remove_com : bool, optional
        Project the centre-of-mass momentum out after each ``O`` step.  Default
        follows ``state.fixed_com``.

        This is a genuine subtlety rather than a nicety.  Langevin noise acts
        on every atom independently, so it *thermalises the centre of mass too*:
        left alone, the system has ``3N`` degrees of freedom, and reporting its
        temperature with the ``3N - 3`` count that
        :class:`~atomlab.md.state.MDState` uses when ``fixed_com`` is set would
        overestimate ``T`` by ``3N/(3N-3)``.  Because the ``O`` step is
        isotropic in mass-weighted velocities, projecting out the (fixed,
        three-dimensional) centre-of-mass subspace commutes with it and leaves
        the remaining ``3N - 3`` degrees of freedom exactly canonical.  So
        removing the drift and counting ``3N - 3`` is consistent; doing neither
        is consistent; doing one without the other is a 1-5% temperature error.
    """

    name = "langevin-baoab"

    def __init__(
        self,
        dt: float,
        temperature: float,
        friction: float,
        rng: np.random.Generator | None = None,
        *,
        remove_com: bool | None = None,
        virial: bool = True,
    ) -> None:
        super().__init__(dt, virial=virial)
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0 K, got {temperature}")
        if friction < 0.0:
            raise ValueError(f"friction must be >= 0 /ps, got {friction}")
        self.temperature = float(temperature)
        self.friction = float(friction)
        self.rng = rng
        self.remove_com = remove_com

        self.c1 = math.exp(-self.friction * self.dt)
        # c2 = sqrt(1 - c1^2); computed via expm1 so that it keeps full relative
        # accuracy for small gamma*dt, where 1 - c1^2 is a difference of nearly
        # equal numbers.
        self.c2 = math.sqrt(-math.expm1(-2.0 * self.friction * self.dt))

    def _generator(self, state: MDState) -> np.random.Generator:
        rng = self.rng if self.rng is not None else state.rng
        if rng is None:
            raise ValueError(
                "Langevin needs a numpy Generator: pass rng= to the integrator "
                "or set state.rng (MDState.from_configuration(seed=...) does)"
            )
        return rng

    def step(self, state: MDState, potential) -> MDState:
        self.initialize(state, potential)
        dt = self.dt
        rng = self._generator(state)
        drop_com = state.fixed_com if self.remove_com is None else self.remove_com

        # B
        self._half_kick(state, 0.5 * dt)
        # A
        state.positions = state.positions + 0.5 * dt * state.velocities
        # O -- exact Ornstein-Uhlenbeck update
        if self.friction > 0.0:
            sigma = np.sqrt(KB * self.temperature / (state.masses * MVV2E))
            noise = rng.standard_normal(state.velocities.shape)
            state.velocities = self.c1 * state.velocities + self.c2 * sigma[:, None] * noise
            if drop_com:
                state.velocities = remove_com_momentum(state.velocities, state.masses)
        # A
        state.positions = state.positions + 0.5 * dt * state.velocities
        # B
        state.refresh(potential, virial=self.virial)
        self._half_kick(state, 0.5 * dt)

        state.step += 1
        state.time += dt
        return state


# --------------------------------------------------------------------------
# Nose-Hoover chains
# --------------------------------------------------------------------------


class NoseHooverChain(Integrator):
    r"""Nose-Hoover chain thermostat (NVT), Suzuki-Yoshida / RESPA propagated.

    A single Nose-Hoover thermostat is *not* ergodic -- for a harmonic
    oscillator it famously samples a torus rather than the canonical
    distribution.  Chaining thermostats (Martyna, Klein and Tuckerman, *J.
    Chem. Phys.* **97**, 2635 (1992)) fixes this: the first thermostat is itself
    thermostatted by a second, and so on, which is why ``chain_length`` defaults
    to 3 and why 1 should be used only to demonstrate the pathology.

    The equations of motion are

    .. math::
        \dot v_i = F_i/m_i - v_{\xi_1} v_i, \qquad
        Q_1 \dot v_{\xi_1} = \sum_i m_i v_i^2 - N_f k_B T - Q_1 v_{\xi_1} v_{\xi_2},
        \qquad
        Q_k \dot v_{\xi_k} = Q_{k-1} v_{\xi_{k-1}}^2 - k_B T - Q_k v_{\xi_k} v_{\xi_{k+1}}

    with masses :math:`Q_1 = N_f k_B T \tau^2` and :math:`Q_k = k_B T \tau^2`.

    **The conserved quantity is the test.**  These equations are not
    Hamiltonian, but they do conserve

    .. math::
        H' = \sum_i \tfrac12 m_i v_i^2 + U + \sum_k \tfrac12 Q_k v_{\xi_k}^2
             + N_f k_B T \xi_1 + k_B T \sum_{k>1} \xi_k

    exactly.  A correct implementation therefore holds :math:`H'` to
    :math:`O(dt^2)` fluctuations with no secular drift, and essentially every
    plausible bug -- wrong chain masses, wrong degree-of-freedom count, a
    non-symmetric propagator, a sign error in the coupling -- breaks it
    visibly.  It is exposed as :meth:`conserved_quantity` for exactly that
    reason.

    Parameters
    ----------
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in K.
    tau : float
        Thermostat time constant in ps.  A sensible choice is 20-100 timesteps;
        much shorter and the chain becomes stiff (raise ``n_respa``), much
        longer and equilibration is slow.
    chain_length : int
        Number of thermostats ``M`` in the chain.  Must be >= 1.
    n_sy : {1, 3, 5, 7}
        Order of the Suzuki-Yoshida decomposition of the thermostat propagator.
        The chain is stiffer than the physical dynamics, so it is integrated to
        higher order at negligible cost -- the sub-steps involve no force
        evaluation.
    n_respa : int
        Number of equal sub-divisions of the half-step used for the chain.
    fixed_com : bool, optional
        Unused directly; the degree-of-freedom count comes from
        ``state.n_dof``, which is where it belongs.
    """

    name = "nose-hoover-chain"

    def __init__(
        self,
        dt: float,
        temperature: float,
        tau: float,
        chain_length: int = 3,
        n_sy: int = 3,
        n_respa: int = 1,
        *,
        virial: bool = True,
    ) -> None:
        super().__init__(dt, virial=virial)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 K, got {temperature}")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0 ps, got {tau}")
        if chain_length < 1:
            raise ValueError(f"chain_length must be >= 1, got {chain_length}")
        if n_respa < 1:
            raise ValueError(f"n_respa must be >= 1, got {n_respa}")
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.chain_length = int(chain_length)
        self.n_sy = int(n_sy)
        self.n_respa = int(n_respa)
        self.weights = suzuki_yoshida_weights(self.n_sy)

    # -- extended-system bookkeeping ---------------------------------------

    @property
    def kt(self) -> float:
        """``k_B T`` in eV."""
        return KB * self.temperature

    def initialize(self, state: MDState, potential) -> MDState:
        super().initialize(state, potential)
        thermo = state.thermostat
        if thermo.get("kind") != self.name or thermo.get("n_dof") != state.n_dof:
            m = self.chain_length
            q = np.full(m, self.kt * self.tau**2)
            # The first chain element carries all N_f degrees of freedom, so its
            # inertia must scale with N_f or the thermostat frequency drifts
            # with system size.
            q[0] *= state.n_dof
            thermo.clear()
            thermo.update(
                kind=self.name,
                xi=np.zeros(m),
                v_xi=np.zeros(m),
                Q=q,
                n_dof=state.n_dof,
                temperature=self.temperature,
            )
        return state

    def _chain_half_step(self, state: MDState) -> None:
        thermo = state.thermostat
        scale = _nhc_propagate(
            two_ke=2.0 * state.kinetic_energy(),
            n_dof_target=state.n_dof,
            kt=self.kt,
            xi=thermo["xi"],
            v_xi=thermo["v_xi"],
            q=thermo["Q"],
            dt_therm=0.5 * self.dt,
            weights=self.weights,
            n_respa=self.n_respa,
        )
        state.velocities *= scale

    def step(self, state: MDState, potential) -> MDState:
        self.initialize(state, potential)
        dt = self.dt

        self._chain_half_step(state)

        self._half_kick(state, 0.5 * dt)
        state.positions = state.positions + dt * state.velocities
        state.refresh(potential, virial=self.virial)
        self._half_kick(state, 0.5 * dt)

        self._chain_half_step(state)

        state.step += 1
        state.time += dt
        return state

    def conserved_quantity(self, state: MDState) -> float:
        """Physical energy plus the chain's kinetic and potential terms, in eV."""
        thermo = state.thermostat
        if thermo.get("kind") != self.name:
            raise RuntimeError("state carries no Nose-Hoover chain; take a step first")
        return state.total_energy() + _chain_energy(
            thermo["xi"], thermo["v_xi"], thermo["Q"], thermo["n_dof"], self.kt
        )

    def thermostat_energy(self, state: MDState) -> float:
        """Energy currently stored in the chain, in eV."""
        thermo = state.thermostat
        return _chain_energy(thermo["xi"], thermo["v_xi"], thermo["Q"], thermo["n_dof"], self.kt)


# --------------------------------------------------------------------------
# MTK barostat
# --------------------------------------------------------------------------


class MTKBarostat(Integrator):
    r"""Isotropic Martyna-Tobias-Klein NPT.

    Extends the Nose-Hoover chain by a single isotropic strain-rate variable
    :math:`v_\epsilon` (units 1/ps) conjugate to :math:`\ln V`, with its own
    thermostat chain:

    .. math::
        \dot r_i &= v_i + v_\epsilon r_i \\
        \dot v_i &= F_i/m_i - \left(1 + \frac{d}{N_f}\right) v_\epsilon v_i
                    - v_{\xi_1} v_i \\
        \dot V &= d\, V v_\epsilon \\
        W \dot v_\epsilon &= d\, V (P_\text{int} - P_\text{ext})
                    + \frac{d}{N_f}\sum_i m_i v_i^2 - W v_{\xi^b_1} v_\epsilon

    with :math:`d = 3` and :math:`W = (N_f + d) k_B T \tau_p^2`.

    The term :math:`\frac{d}{N_f}\sum_i m_i v_i^2`, and the matching
    :math:`d/N_f` in the velocity equation, are the MTK correction to the
    earlier Hoover NPT equations.  They look like a small bookkeeping detail and
    are not: without them the sampled distribution is off by :math:`O(1/N)`,
    which is invisible at :math:`N = 10^5` and quite visible at the few-hundred
    atoms this project runs.  With them, the dynamics conserves

    .. math::
        H' = KE + U + P_\text{ext} V + \tfrac12 W v_\epsilon^2
             + (\text{particle chain}) + (\text{barostat chain})

    which is again the sharpest available test and is exposed as
    :meth:`conserved_quantity`.

    The position/cell update uses the exact flow of
    :math:`\dot r = v + v_\epsilon r` at fixed ``v``, i.e.
    ``r <- r e^{2x} + dt\, v\, e^{x} \sinh(x)/x`` with
    :math:`x = v_\epsilon dt/2`, rather than a Taylor expansion; that keeps the
    scheme measure-preserving and reversible.

    Parameters
    ----------
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in K.
    pressure : float
        Target pressure in **bar** (the metal-unit convention; 1 atm ~ 1.01325
        bar).  Converted internally to eV/A^3.
    tau_t, tau_p : float
        Thermostat and barostat time constants in ps.  ``tau_p`` should be
        several times ``tau_t`` so the volume relaxes slowly compared with the
        temperature.
    chain_length, n_sy, n_respa : int
        As for :class:`NoseHooverChain`; the same settings are used for both
        chains.
    """

    name = "mtk-npt"

    def __init__(
        self,
        dt: float,
        temperature: float,
        pressure: float,
        tau_t: float,
        tau_p: float,
        *,
        chain_length: int = 3,
        n_sy: int = 3,
        n_respa: int = 1,
    ) -> None:
        super().__init__(dt, virial=True)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 K, got {temperature}")
        if tau_t <= 0.0 or tau_p <= 0.0:
            raise ValueError("tau_t and tau_p must be > 0 ps")
        if chain_length < 1:
            raise ValueError(f"chain_length must be >= 1, got {chain_length}")
        self.temperature = float(temperature)
        self.pressure_bar = float(pressure)
        #: Target pressure in eV/A^3, the internal stress unit.
        self.pressure = self.pressure_bar * BAR_TO_EV_A3
        self.tau_t = float(tau_t)
        self.tau_p = float(tau_p)
        self.chain_length = int(chain_length)
        self.n_sy = int(n_sy)
        self.n_respa = int(n_respa)
        self.weights = suzuki_yoshida_weights(self.n_sy)

    @property
    def kt(self) -> float:
        """``k_B T`` in eV."""
        return KB * self.temperature

    # -- extended-system bookkeeping ---------------------------------------

    def initialize(self, state: MDState, potential) -> MDState:
        super().initialize(state, potential)
        if not np.asarray(state.pbc).all():
            raise ValueError("MTKBarostat requires a fully periodic cell")
        thermo = state.thermostat
        if thermo.get("kind") != self.name or thermo.get("n_dof") != state.n_dof:
            m = self.chain_length
            q = np.full(m, self.kt * self.tau_t**2)
            q[0] *= state.n_dof
            thermo.clear()
            thermo.update(
                kind=self.name,
                xi=np.zeros(m),
                v_xi=np.zeros(m),
                Q=q,
                n_dof=state.n_dof,
                temperature=self.temperature,
            )
        baro = state.barostat
        if baro.get("kind") != self.name or baro.get("n_dof") != state.n_dof:
            m = self.chain_length
            baro.clear()
            baro.update(
                kind=self.name,
                v_eps=0.0,
                # W = (N_f + d) kT tau_p^2 is the MTK barostat inertia; the
                # (N_f + d) makes the volume-fluctuation timescale tau_p
                # independent of system size.
                W=(state.n_dof + 3) * self.kt * self.tau_p**2,
                xi=np.zeros(m),
                v_xi=np.zeros(m),
                # The barostat chain thermostats a single degree of freedom,
                # so all its masses are kT tau_t^2 with no N_f factor.
                Q=np.full(m, self.kt * self.tau_t**2),
                n_dof=state.n_dof,
            )
        return state

    def _internal_pressure(self, state: MDState) -> float:
        """Instantaneous pressure in eV/A^3, kinetic plus virial."""
        return state.pressure(unit="eV/A^3")

    def _g_eps(self, state: MDState) -> float:
        """``dv_eps/dt`` from the pressure imbalance, in 1/ps^2."""
        volume = state.volume
        two_ke = 2.0 * state.kinetic_energy()
        p_int = self._internal_pressure(state)
        return (
            3.0 * volume * (p_int - self.pressure) + (3.0 / state.n_dof) * two_ke
        ) / state.barostat["W"]

    def _chains_half_step(self, state: MDState) -> None:
        thermo, baro = state.thermostat, state.barostat
        scale = _nhc_propagate(
            two_ke=2.0 * state.kinetic_energy(),
            n_dof_target=state.n_dof,
            kt=self.kt,
            xi=thermo["xi"],
            v_xi=thermo["v_xi"],
            q=thermo["Q"],
            dt_therm=0.5 * self.dt,
            weights=self.weights,
            n_respa=self.n_respa,
        )
        state.velocities *= scale

        scale_b = _nhc_propagate(
            two_ke=baro["W"] * baro["v_eps"] ** 2,
            n_dof_target=1,
            kt=self.kt,
            xi=baro["xi"],
            v_xi=baro["v_xi"],
            q=baro["Q"],
            dt_therm=0.5 * self.dt,
            weights=self.weights,
            n_respa=self.n_respa,
        )
        baro["v_eps"] *= scale_b

    def _velocity_half_step(self, state: MDState) -> None:
        """Half kick combined with the exact barostat drag ``exp(-a1 t)``."""
        dt = self.dt
        a1 = (1.0 + 3.0 / state.n_dof) * state.barostat["v_eps"]
        x = 0.25 * a1 * dt
        acc = _accelerations(state.forces, state.masses)
        state.velocities = state.velocities * math.exp(-2.0 * x) + (
            0.5 * dt * math.exp(-x) * _sinhx_over_x(x)
        ) * acc

    def step(self, state: MDState, potential) -> MDState:
        self.initialize(state, potential)
        dt = self.dt
        baro = state.barostat

        self._chains_half_step(state)

        baro["v_eps"] += 0.5 * dt * self._g_eps(state)
        self._velocity_half_step(state)

        # exact flow of dr/dt = v + v_eps r at fixed v, and dcell/dt = v_eps cell
        x = 0.5 * baro["v_eps"] * dt
        expx = math.exp(x)
        state.positions = state.positions * (expx * expx) + (
            dt * expx * _sinhx_over_x(x)
        ) * state.velocities
        state.cell = state.cell * (expx * expx)
        state.refresh(potential, virial=True)

        self._velocity_half_step(state)
        baro["v_eps"] += 0.5 * dt * self._g_eps(state)

        self._chains_half_step(state)

        state.step += 1
        state.time += dt
        return state

    def conserved_quantity(self, state: MDState) -> float:
        """MTK conserved quantity in eV (physical + PV + barostat + both chains)."""
        thermo, baro = state.thermostat, state.barostat
        if baro.get("kind") != self.name:
            raise RuntimeError("state carries no MTK barostat; take a step first")
        h = state.total_energy()
        h += self.pressure * state.volume
        h += 0.5 * baro["W"] * baro["v_eps"] ** 2
        h += _chain_energy(thermo["xi"], thermo["v_xi"], thermo["Q"], thermo["n_dof"], self.kt)
        h += _chain_energy(baro["xi"], baro["v_xi"], baro["Q"], 1, self.kt)
        return h

    @staticmethod
    def instantaneous_pressure(state: MDState, *, unit: str = "bar") -> float:
        """Convenience passthrough to :meth:`MDState.pressure`."""
        return state.pressure(unit=unit)


# --------------------------------------------------------------------------
# Berendsen -- the negative control
# --------------------------------------------------------------------------


class BerendsenThermostat(Integrator):
    r"""Velocity Verlet with Berendsen velocity rescaling.

    .. warning::
       **This does not sample the canonical ensemble.**  It is provided as a
       negative control and must never be used to generate reference samples.

    Each step the velocities are multiplied by

    .. math::  \lambda = \sqrt{1 + \frac{dt}{\tau}\left(\frac{T_0}{T} - 1\right)}

    which drives the *mean* kinetic energy to its target exponentially with time
    constant :math:`\tau/2`.  Nothing in that construction controls the
    *distribution*: the kinetic-energy fluctuations are suppressed by roughly
    :math:`\tau/dt`, so the sampled ensemble interpolates between canonical
    (:math:`\tau \to \infty`) and isokinetic (:math:`\tau \to dt`), and is
    neither.  Since the heat capacity is a fluctuation, any observable that is a
    response function is simply wrong.

    Worse, and more instructive, is the **flying ice cube**.  Rescaling is
    uniform, so it drains energy from every mode in proportion to the energy the
    mode already has; over a long run energy leaks systematically from the
    high-frequency internal modes into the three zero-frequency
    centre-of-mass modes, until the system translates rigidly through the box
    while its internal temperature reads correctly.  Every standard
    diagnostic -- mean temperature, total energy, even the pressure -- looks
    fine.  The physics is destroyed.

    That failure mode is the reason this class exists in a repository about
    metrics that look fine while the physics is wrong.  ``remove_com=False``
    reproduces it; the default ``True`` suppresses it, which is what production
    codes do and which is exactly the sort of patch that hides a broken method
    behind a healthy-looking metric.

    Parameters
    ----------
    dt : float
        Timestep in ps.
    temperature : float
        Target temperature in K.
    tau : float
        Coupling time constant in ps.  ``tau = dt`` gives naive velocity
        rescaling (isokinetic); large ``tau`` approaches NVE.
    remove_com : bool
        Remove the centre-of-mass drift each step.  Set False to expose the
        flying ice cube.
    """

    name = "berendsen"

    def __init__(
        self,
        dt: float,
        temperature: float,
        tau: float,
        *,
        remove_com: bool = True,
        virial: bool = True,
    ) -> None:
        super().__init__(dt, virial=virial)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 K, got {temperature}")
        if tau < dt:
            raise ValueError(f"tau must be >= dt; got tau={tau}, dt={dt}")
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.remove_com = bool(remove_com)

    def step(self, state: MDState, potential) -> MDState:
        self.initialize(state, potential)
        dt = self.dt

        self._half_kick(state, 0.5 * dt)
        state.positions = state.positions + dt * state.velocities
        state.refresh(potential, virial=self.virial)
        self._half_kick(state, 0.5 * dt)

        current = state.temperature()
        if current > 0.0:
            lam_sq = 1.0 + (dt / self.tau) * (self.temperature / current - 1.0)
            # lam_sq can go negative for tau ~ dt and a very cold start; clamping
            # is standard practice and is itself a symptom of the method having
            # no underlying dynamics to be consistent with.
            state.velocities *= math.sqrt(max(lam_sq, 0.0))
        if self.remove_com:
            state.zero_com_momentum()

        state.step += 1
        state.time += dt
        return state
