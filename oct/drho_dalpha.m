function d = drho_dalpha(t, alpha, psi, gamma, omega)
  d = sin(2.*pi.*sigmoid((t .- psi)./730, gamma));
end
