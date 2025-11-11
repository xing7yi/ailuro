#pragma once

#include "IsotropicLinearHardeningStressUpdate.h"

/**
 * This class uses the Discrete material in a radial return isotropic plasticity
 * model.  This class is one of the basic radial return constitutive models;
 * more complex constitutive models combine creep and plasticity.
 *
 * This class models Voce law hardening by using the relation
 * \f$ \sigma = \sigma_y + R * \epsilon + Q * (1 - \exp(-b * \epsilon))\f$
 * where \f$ \sigma_y \f$ is the yield stress. 
 */

template <bool is_ad>
class IsotropicVoceLawHardeningStressUpdateTempl
  : public IsotropicLinearHardeningStressUpdateTempl<is_ad>
{
public:
  static InputParameters validParams();

  IsotropicVoceLawHardeningStressUpdateTempl(const InputParameters & parameters);

  using Material::_qp;
  using RadialReturnStressUpdateTempl<is_ad>::_base_name;  
  using RadialReturnStressUpdateTempl<is_ad>::_three_shear_modulus;

protected:

  virtual void computeStressInitialize(
      const GenericReal<is_ad> & effective_trial_stress,
      const GenericRankFourTensor<is_ad> & elasticity_tensor) override;

  virtual GenericReal<is_ad> computeHardeningValue(const GenericReal<is_ad> & scalar) override;

  ///@{ Voce law hardening coefficients
  const Real & _q;
  const Real & _b;  // Reference from input parameter (controllable)
  Real _b_value;    // Working copy for computed default value
  GenericReal<is_ad> _youngs_modulus;  // Computed from elasticity tensor
  ///@}

  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_slope;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_variable;
  using IsotropicLinearHardeningStressUpdateTempl<is_ad>::_hardening_variable_old;

  GenericReal<is_ad> getIsotropicLameLambda(const GenericRankFourTensor<is_ad> & elasticity_tensor);
};

typedef IsotropicVoceLawHardeningStressUpdateTempl<false> IsotropicVoceLawHardeningStressUpdate;
typedef IsotropicVoceLawHardeningStressUpdateTempl<true> ADIsotropicVoceLawHardeningStressUpdate;
