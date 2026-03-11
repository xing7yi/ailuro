#include "IsotropicVoceLawHardeningStressUpdate.h"
using MetaPhysicL::exp;
#include "ElasticityTensorTools.h"

registerMooseObject("SolidMechanicsApp", ADIsotropicVoceLawHardeningStressUpdate);
registerMooseObject("SolidMechanicsApp", IsotropicVoceLawHardeningStressUpdate);


template <bool is_ad>
InputParameters
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::validParams()
{
  InputParameters params = IsotropicLinearHardeningStressUpdateTempl<is_ad>::validParams();
  params.addClassDescription("This class uses the discrete material in a radial return isotropic "
                             "plasticity Voce law hardening model."
                             " This class can be used in conjunction with other creep and "
                             "plasticity materials for more complex simulations.");

  // Voce law hardening specific parameters
  params.addParam<Real>("q", 0.0, "Saturation value for isotropic hardening (Q in Voce model)");
  params.addParam<Real>("b", 0.0, "Rate constant for isotropic hardening (b in Voce model). ");
  params.declareControllable("q b");
  return params;
}


template <bool is_ad>
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::IsotropicVoceLawHardeningStressUpdateTempl(
    const InputParameters & parameters)
  : IsotropicLinearHardeningStressUpdateTempl<is_ad>(parameters),
    _q(this->template getParam<Real>("q")),
    _b(this->template getParam<Real>("b"))
    // _b_value(_b),  // Initialize working copy with input value
    // _youngs_modulus(0.0)
{
}

template <bool is_ad>
void
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::computeStressInitialize(
    const GenericReal<is_ad> & effective_trial_stress,
    const GenericRankFourTensor<is_ad> & elasticity_tensor)
{
  // Call parent class implementation first
  IsotropicLinearHardeningStressUpdateTempl<is_ad>::computeStressInitialize(
      effective_trial_stress, elasticity_tensor);

  // // Compute Young's modulus from elasticity tensor
  // // (similar to IsotropicPowerLawHardeningStressUpdate)
  // const GenericReal<is_ad> lambda = getIsotropicLameLambda(elasticity_tensor);
  // const GenericReal<is_ad> shear_modulus = _three_shear_modulus / 3.0;
  // _youngs_modulus = shear_modulus * (3.0 * lambda + 2.0 * shear_modulus) / (lambda + shear_modulus);

  // // If b not user-specified (or is zero/default) but Q > 0, compute b = (E - H) / Q
  // if ((_b_value <= 0.0 || !this->isParamSetByUser("b")) && _q > 0.0)
  // {
  //   // Extract raw value from GenericReal (handles both AD and non-AD cases)
  //   _b_value = MetaPhysicL::raw_value(_youngs_modulus - this->_hardening_slope) / _q;
  // }
}

template <bool is_ad>
GenericReal<is_ad>
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::computeHardeningValue(
    const GenericReal<is_ad> & scalar)
{
  // EXACT Voce formula using TOTAL accumulated plastic strain
  // scalar = plastic strain increment in current Newton iteration
  // Total plastic strain = old + increment
  const GenericReal<is_ad> total_plastic_strain = 
      this->_effective_inelastic_strain_old[_qp] + scalar;
  
  // Update Voce hardening variable for output/monitoring
  _hardening_variable[_qp] = _q * (1.0 - exp(-_b * total_plastic_strain));
  
  // Return EXACT hardening value: H = R*epsilon_total + Q*(1 - exp(-b*epsilon_total))
  // This differs from MOOSE official which uses first-order approximation
  return (_hardening_slope * total_plastic_strain + _hardening_variable[_qp]);
}

// template <bool is_ad>
// GenericReal<is_ad>
// IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::getIsotropicLameLambda(
//     const GenericRankFourTensor<is_ad> & elasticity_tensor)
// {
//   const GenericReal<is_ad> lame_lambda = elasticity_tensor(0, 0, 1, 1);

//   if (this->_mesh.dimension() == 3 &&
//       MetaPhysicL::raw_value(lame_lambda) != MetaPhysicL::raw_value(elasticity_tensor(1, 1, 2, 2)))
//     mooseError(
//         "Check to ensure that your Elasticity Tensor is truly Isotropic: different lambda values");
//   return lame_lambda;
// }

template class IsotropicVoceLawHardeningStressUpdateTempl<false>;
template class IsotropicVoceLawHardeningStressUpdateTempl<true>;
