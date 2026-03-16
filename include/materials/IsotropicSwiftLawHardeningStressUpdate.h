#pragma once

#include "IsotropicLinearHardeningStressUpdate.h"

/**
 * Isotropic plasticity with Swift power law hardening:
 *
 *   sigma_p = sigma_0 * (1 + eps_p / eps_0)^n
 *
 * where sigma_0 is the yield strength (flow stress at eps_p = 0).
 *
 * Decomposed in MOOSE convention as:
 *   flow stress = yield_stress + hardening_variable
 *              = sigma_0       + sigma_0 * [(1 + eps_p/eps_0)^n - 1]
 *
 * Uses radial return mapping (J2 plasticity).
 */
template <bool is_ad>
class IsotropicSwiftLawHardeningStressUpdateTempl
  : public IsotropicLinearHardeningStressUpdateTempl<is_ad>
{
public:
  static InputParameters validParams();

  IsotropicSwiftLawHardeningStressUpdateTempl(const InputParameters & parameters);

  using Material::_qp;
  using RadialReturnStressUpdateTempl<is_ad>::_three_shear_modulus;
  using RadialReturnStressUpdateTempl<is_ad>::_effective_inelastic_strain_old;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_yield_stress;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_slope;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_variable;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_variable_old;

protected:
  /// Set _yield_stress = sigma_0 (called from computeStressInitialize in parent)
  // virtual void computeYieldStress(const GenericRankFourTensor<is_ad> & elasticity_tensor) override;

  /// Hardening above initial yield: H = sigma_0 * [(1 + eps_p/eps_0)^n - 1]
  virtual GenericReal<is_ad> computeHardeningValue(const GenericReal<is_ad> & scalar) override;

  /// dH/d(eps_p) = sigma_0 * n / eps_0 * (1 + eps_p/eps_0)^(n-1)
  virtual GenericReal<is_ad> computeHardeningDerivative(const GenericReal<is_ad> & scalar) override;

  // --- Swift law parameters ---
  const Real _sigma_0; ///< Initial yield stress (= flow stress at eps_p = 0)
  const Real _eps_0;   ///< Reference plastic strain
  const Real _n;       ///< Hardening exponent
};

typedef IsotropicSwiftLawHardeningStressUpdateTempl<false> IsotropicSwiftLawHardeningStressUpdate;
typedef IsotropicSwiftLawHardeningStressUpdateTempl<true> ADIsotropicSwiftLawHardeningStressUpdate;