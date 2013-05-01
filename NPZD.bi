/**
 * NPZD model.
 * 
 * @author Emlyn Jones <emlyn.jones@csiro.au>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * 
 * $Rev$
 * $Date$
 */
model NPZD {
  /* prescribed constants */
  const Pt = 10.0  // P timescale
  const Zt = 30.0  // Z timescale
  const Q10 = 2.0
  const Trf = 20.0     // reference temperature (10 deg. C)
  const PaC = 1.0      // chlorophyl absorption
  const PQf = 1200.00  // photosynthetic efficiency
  const PNC = 0.18     // maximum N:C (Redfield)
  const Zex = 2.0      // functional response exponent

  /* forcings */
  input BCP   // phytoplankton boundary condition
  input BCZ   // zooplankton boundary condition
  input BCN   // nitrogen boundary condition
  input BCD   // detritus boundary condition
  input FT    // water temperature
  input FE    // surface irradience
  input FMLD  // mixed layer depth
  input FMLC  // mixed layer Change (t - t-1)
  input FMIX  // mixing and exchange

  /* parameters */
  param Kw   // light attenuation, water
  param KCh  // light attenuation, chl-a
  param Dsi  // detrital sinking rate
  param ZgD  // Z grazing inefficiency
  param PDF  // P diversity factor
  param ZDF  // Z diversity factor

  /* state variables */
  state EZ    // light at depth
  state Chla  // chlorophyl-a
  state P     // phytoplankton
  state Z     // zooplankton
  state D     // detritus
  state N     // nutrient

  /* rate process autoregressives */
  state PgC  // maximum growth rate for C
  state PCh  // maximum growth rate for Chl-a:C
  state PRN  // min/max N:C
  state ASN  // a*n = PaN*PgR
  state Zin  // max Z ingestion rate
  state ZCl  // zooplankton clearance rate
  state ZgE  // zooplankton grazing efficiency
  state Dre  // remineralisation rate
  state ZmQ  // Z quadratic mortality term

  /* rate process noise terms */
  noise rPgC, rPCh, rPRN, rASN, rZin, rZCl, rZgE, rDre, rZmQ

  /* rate process long-term means */
  param muPgC, muPCh, muPRN, muASN, muZin, muZCl, muZgE, muDre, muZmQ	

  /* prescribed rate process long-term standard deviations */ 
  const thetaPgC = 0.63
  const thetaPCh = 0.37
  const thetaPRN = 0.3
  const thetaASN = 1.0
  const thetaZin = 0.7
  const thetaZCl = 1.3
  const thetaZgE = 0.25
  const thetaDre = 0.5
  const thetaZmQ = 1.0  // 0.5?

  /* rate process noise term means */
  inline sigmaPgC = sqrt(log(1.0 + (2.0*Pt - 1.0)*PDF*PDF*(exp(thetaPgC*thetaPgC) - 1.0)))
  inline sigmaPCh = sqrt(log(1.0 + (2.0*Pt - 1.0)*PDF*PDF*(exp(thetaPCh*thetaPCh) - 1.0)))
  inline sigmaPRN = sqrt(log(1.0 + (2.0*Pt - 1.0)*PDF*PDF*(exp(thetaPRN*thetaPRN) - 1.0)))
  inline sigmaASN = sqrt(log(1.0 + (2.0*Pt - 1.0)*PDF*PDF*(exp(thetaASN*thetaASN) - 1.0)))
  inline sigmaZin = sqrt(log(1.0 + (2.0*Zt - 1.0)*ZDF*ZDF*(exp(thetaZin*thetaZin) - 1.0)))
  inline sigmaZCl = sqrt(log(1.0 + (2.0*Zt - 1.0)*ZDF*ZDF*(exp(thetaZCl*thetaZCl) - 1.0)))
  inline sigmaZgE = sqrt(log(1.0 + (2.0*Zt - 1.0)*ZDF*ZDF*(exp(thetaZgE*thetaZgE) - 1.0)))
  inline sigmaDre = sqrt(log(1.0 + (2.0*Zt - 1.0)*ZDF*ZDF*(exp(thetaDre*thetaDre) - 1.0)))
  inline sigmaZmQ = sqrt(log(1.0 + (2.0*Zt - 1.0)*ZDF*ZDF*(exp(thetaZmQ*thetaZmQ) - 1.0)))

  /* rate process noise term standard deviations */
  inline gammaPgC = log(muPgC) + pow(thetaPgC, 2.0)/2.0 - pow(sigmaPgC, 2.0)/2.0
  inline gammaPCh = log(muPCh) + pow(thetaPCh, 2.0)/2.0 - pow(sigmaPCh, 2.0)/2.0
  inline gammaPRN = log(muPRN) + pow(thetaPRN, 2.0)/2.0 - pow(sigmaPRN, 2.0)/2.0
  inline gammaASN = log(muASN) + pow(thetaASN, 2.0)/2.0 - pow(sigmaASN, 2.0)/2.0
  inline gammaZin = log(muZin) + pow(thetaZin, 2.0)/2.0 - pow(sigmaZin, 2.0)/2.0
  inline gammaZCl = log(muZCl) + pow(thetaZCl, 2.0)/2.0 - pow(sigmaZCl, 2.0)/2.0
  inline gammaZgE = log(muZgE) + pow(thetaZgE, 2.0)/2.0 - pow(sigmaZgE, 2.0)/2.0
  inline gammaDre = log(muDre) + pow(thetaDre, 2.0)/2.0 - pow(sigmaDre, 2.0)/2.0
  inline gammaZmQ = log(muZmQ) + pow(thetaZmQ, 2.0)/2.0 - pow(sigmaZmQ, 2.0)/2.0

  /* observations */
  obs N_obs, Chla_obs

  /* prior distribution over parameters */
  sub parameter {
    Kw ~ log_normal(log(0.03), 0.2)
    KCh ~ log_normal(log(0.02), 0.3)
    Dsi ~ normal(5.0, 1.0)
    ZgD ~ log_normal(log(0.5), 0.1)
    PDF ~ log_normal(log(0.2), 0.4)
    ZDF ~ log_normal(log(0.2), 0.4)

    muPgC ~ log_normal(log(1.2), thetaPgC)
    muPCh ~ log_normal(log(0.03), thetaPCh)
    muPRN ~ log_normal(log(0.25), thetaPRN)
    muASN ~ log_normal(log(0.3), thetaASN)
    muZin ~ log_normal(log(4.7), thetaZin)
    muZCl ~ log_normal(log(0.2), thetaZCl)
    muZgE ~ log_normal(log(0.32), thetaZgE)
    muDre ~ log_normal(log(0.1), thetaDre)
    muZmQ ~ log_normal(log(0.01), thetaZmQ)
  }

  const prop_std = 0.1
  sub proposal_parameter {
    Kw ~ log_normal(log(Kw), 0.2*prop_std)
    KCh ~ log_normal(log(KCh), 0.3*prop_std)
    Dsi ~ gaussian(Dsi, 1.0*prop_std)
    ZgD ~ log_normal(log(ZgD), 0.1*prop_std)
    PDF ~ log_normal(log(PDF), 0.4*prop_std)
    ZDF ~ log_normal(log(ZDF), 0.4*prop_std)

    muPgC ~ log_normal(log(muPgC), thetaPgC*prop_std)
    muPCh ~ log_normal(log(muPCh), thetaPCh*prop_std)
    muPRN ~ log_normal(log(muPRN), thetaPRN*prop_std)
    muASN ~ log_normal(log(muASN), thetaASN*prop_std)
    muZin ~ log_normal(log(muZin), thetaZin*prop_std)
    muZCl ~ log_normal(log(muZCl), thetaZCl*prop_std)
    muZgE ~ log_normal(log(muZgE), thetaZgE*prop_std)
    muDre ~ log_normal(log(muDre), thetaDre*prop_std)
    muZmQ ~ log_normal(log(muZmQ), thetaZmQ*prop_std)
  }

  /* prior distribution over initial conditions, given parameters */
  sub initial {
    PgC ~ log_normal(log(muPgC), sigmaPgC)
    PCh ~ log_normal(log(muPCh), sigmaPCh)
    PRN ~ log_normal(log(muPRN), sigmaPRN)
    ASN ~ log_normal(log(muASN), sigmaASN)
    Zin ~ log_normal(log(muZin), sigmaZin)
    ZCl ~ log_normal(log(muZCl), sigmaZCl)
    ZgE ~ log_normal(log(muZgE), sigmaZgE)
    Dre ~ log_normal(log(muDre), sigmaDre)
    ZmQ ~ log_normal(log(muZmQ), sigmaZmQ)

    P ~ log_normal(log(1.7763), 0.3)
    Z ~ log_normal(log(3.7753), 0.3)
    D ~ log_normal(log(2.9182), 0.3)
    N ~ log_normal(log(176.39), 0.3)

    Chla ~ log_normal(log(0.6469), 0.5)
    EZ ~ log_normal(log(1.1), 1.0)
  }

  /* transition distribution */
  sub transition(delta = 1.0) {
    /* processes */
    inline Tc = pow(Q10, (FT - Trf)/10.0)
    inline Kdz = (Kw + KCh*Chla)*FMLD // total light attenation, water+Chl-a
    inline Zgs = pow(ZCl*P/Zin, Zex)
    inline Zgr = Z*Zin*Tc*Zgs/(1.0 + Zgs)
    inline PgT = PgC*Tc
    inline PaQ = PaC*KCh*PQf
    inline PfE = 1.0 - exp(-PaQ*PCh*EZ/PgC)
    inline PfN = N/(1.0 + ASN*N/PgT)
    inline Pg = PgT*PfE*PfN/(PfN + PfE) // phytoplankton growth rate
    inline Zm = ZmQ*Z*Z

    /* autoregressive noise terms */
    rPgC ~ log_normal(gammaPgC, sigmaPgC)
    rPCh ~ log_normal(gammaPCh, sigmaPCh)
    rPRN ~ log_normal(gammaPRN, sigmaPRN)
    rASN ~ log_normal(gammaASN, sigmaASN)
    rZin ~ log_normal(gammaZin, sigmaZin)
    rZCl ~ log_normal(gammaZCl, sigmaZCl)
    rZgE ~ log_normal(gammaZgE, sigmaZgE)
    rDre ~ log_normal(gammaDre, sigmaDre)
    rZmQ ~ log_normal(gammaZmQ, sigmaZmQ)

    /* autoregressives */
    PgC <- PgC*(1.0 - 1.0/Pt) + rPgC/Pt
    PCh <- PCh*(1.0 - 1.0/Pt) + rPCh/Pt
    PRN <- PRN*(1.0 - 1.0/Pt) + rPRN/Pt
    ASN <- ASN*(1.0 - 1.0/Pt) + rASN/Pt
    Zin <- Zin*(1.0 - 1.0/Zt) + rZin/Zt
    ZCl <- ZCl*(1.0 - 1.0/Zt) + rZCl/Zt
    ZgE <- ZgE*(1.0 - 1.0/Zt) + rZgE/Zt
    Dre <- Dre*(1.0 - 1.0/Zt) + rDre/Zt
    ZmQ <- ZmQ*(1.0 - 1.0/Zt) + rZmQ/Zt

    /* light */
    EZ <- FE*(1.0 - exp(-Kdz))/Kdz

    /* differential system */
    ode(h = 0.1, atoler = 1.0e-4, rtoler = 1.0e-4, alg = 'RK4(3)') {
      dP/dt = Pg*P - Zgr + FMIX*(BCP - P)
      dZ/dt = Zgr*ZgE - Zm + FMLC/FMLD*(BCZ - Z)
      dD/dt = (1.0 - ZgE)*ZgD*Zgr + Zm - Dre*D - Dsi*D/FMLD + FMIX*(BCD - D)
      dN/dt = -Pg*P + (1.0 - ZgE)*(1.0 - ZgD)*Zgr + Dre*D + FMIX*(BCN - N)
    }

    /* chlorophyl-a */
    Chla <- Tc*P*(PCh/PNC)*PfN/(PRN*PfE + PfN)
  }

  /* observation model */
  sub observation {
    N_obs ~ log_normal(log(N), 0.2)
    Chla_obs ~ log_normal(log(Chla), 0.5)
  }
}
