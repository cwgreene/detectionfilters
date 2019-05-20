import numpy

def dynamics(particles, a, dt, config):
    p_x = particles[:,0,:]
    p_v = particles[:,1,:]

    a_drift = a + numpy.random.normal(0, config["ACCEL_DRIFT"], size=p_x.shape)
    x_drift = .5 * a_drift * dt ** 2 + p_v * dt + p_x
    v_drift = a_drift * dt + p_v

    return numpy.stack((x_drift, v_drift), 1)

def dynamics2(particles, a, dt, config):
    p_x = particles[:,0,:]
    p_xp = particles[:,1,:]
    
    a_drift = a + numpy.random.normal(0, config["ACCEL_DRIFT"], size=p_x.shape)
    p_xn1 = a_drift * dt**2 + 2*p_x - p_xp

    return numpy.stack((p_xn1, p_x), 1)


