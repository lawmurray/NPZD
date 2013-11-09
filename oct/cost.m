function [val, dval] = cost(theta, s, t, data)
    val = meansq(data .- mu(s, t, theta));
    dval = -2.0.*mean((data .- mu(s, t, theta)).*dmu_dtheta(s, t, theta));
    dval = dval';
end
