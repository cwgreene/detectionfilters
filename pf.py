import numpy
import draw
import dynamics

DIMENSION = 2
FACTORS = 2 # Position, velocity

MODEL_SIGMA = .3
ACCEL_DRIFT = .1
# circular particle
def signal_x(t):
    return numpy.stack((numpy.cos(t/2), numpy.sin(t/2)), 1)

def signal_v(t):
    return numpy.stack((-.5*numpy.sin(t/2), .5*numpy.cos(t/2)), 1)

def signal_a(t):
    return numpy.stack((-(.5**2)*numpy.cos(t/2), -(.5**2)*numpy.sin(t/2)), 1)
def likelihood(particles, measurement, weights):
    p_x = particles[:,0,:]
    mu = p_x-measurement
    l = numpy.exp(-numpy.einsum("ij,ij->i", mu,mu)/(2*MODEL_SIGMA))
    return weights * l

def resample(particles, weights):
    return particles[numpy.random.choice(len(particles), len(particles), p=weights)]

def particle_filter(n, control, measurements, true_pos, true_vel, dt, dynamics, config):
    p = numpy.random.uniform(-1, 1, size=(n, FACTORS, DIMENSION))
    p[:,0,:] # for verlet assume initially stationary.
    w = 1.0/n  + numpy.zeros(n)
    for u, m, tx, tv in zip(control, measurements, true_pos, true_vel):
        # Do dynamics
        drift_p = dynamics(p, u, dt, config)
        # compute weight based on measurements and existing weights
        w = likelihood(drift_p, m, w)
        print(numpy.sum(w))
        if numpy.sum(w) == 0:
            print("boom")
            return
        w = w / numpy.sum(w)
        # resample
        p = resample(drift_p, w)
        draw.pyframe(p[:,0,:], p[:,1,:], tx, tv)
draw.init()

dt = .1
t = numpy.arange(0,1000, dt)
x = signal_x(t)
v = signal_v(t)
a = signal_a(t)

particle_filter(20000,
    a,
    x + numpy.random.normal(0, .1, size=x.shape),
    x, v, dt, dynamics.dynamics, {"ACCEL_DRIFT": ACCEL_DRIFT})
