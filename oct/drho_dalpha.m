function d = drho_dalpha(t, alpha, psi, gamma, omega)
  d = sin(2.*pi.*sigmoid((fmod(t .- psi .+ 365, 365))./365, gamma));
end
