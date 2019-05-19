import numpy
import draw

DIMENSION = 2
FACTORS = 2 # Position, velocity

MODEL_SIGMA = 1
ACCEL_DRIFT = 1
# circular particle
def signal_x(t):
    return numpy.stack((numpy.cos(t), numpy.sin(t)), 1)

def signal_a(t):
    return numpy.stack((-numpy.cos(t), -numpy.sin(t)), 1)

def dynamics(particles, a, dt):
    p_x = particles[:,0,:]
    p_v = particles[:,1,:]

    a_drift = a + numpy.random.normal(0, ACCEL_DRIFT, size=p_x.shape)
    x_drift = .5 * a_drift * dt ** 2 + p_v * dt + p_x
    v_drift = a * dt + p_v

    return numpy.stack((x_drift, v_drift), 1)

def dynamics2(particles, a, dt):
    p_x = particles[:,0,:]
    p_xp = particles[:,1,:]
    
    a_drift = a + numpy.random.normal(0, ACCEL_DRIFT, size=p_x.shape)
    p_xn1 = a_drift * dt**2 + 2*p_x - p_xp

    return numpy.stack((p_xn1, p_x), 1)

def likelihood(particles, measurement, weights):
    p_x = particles[:,0,:]
    mu = p_x-measurement
    l = numpy.exp(-numpy.einsum("ij,ij->i", mu,mu)/(2*MODEL_SIGMA))
    return weights * l

def resample(particles, weights):
    return particles[numpy.random.choice(len(particles), len(particles), p=weights)]

def particle_filter(n, control, measurements, truth, dt):
    p = numpy.random.uniform(-1, 1, size=(n, FACTORS, DIMENSION))
    w = 1.0/n  + numpy.zeros(n)
    for u, m, t in zip(control, measurements, truth):
        # Do dynamics
        drift_p = dynamics2(p, u, dt)
        # compute weight based on measurements and existing weights
        w = likelihood(drift_p, m, w)
        print(numpy.sum(w))
        if numpy.sum(w) == 0:
            print("boom")
            return
        w = w / numpy.sum(w)
        # resample
        p = resample(drift_p, w)
        draw.pyframe(p[:,0,:], t)
draw.init()

dt = .1
t = numpy.arange(0,1000, dt)
x = signal_x(t)
a = signal_a(t)

particle_filter(20000, a, x + numpy.random.normal(0,.1, size=x.shape), x, dt)
