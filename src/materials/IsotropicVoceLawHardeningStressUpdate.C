#include "IsotropicVoceLawHardeningStressUpdate.h"
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
  params.addParam<Real>("b", 0.0, "Rate constant for isotropic hardening (b in Voce model)"); 
  params.declareControllable("q b");
  return params;
}


template <bool is_ad>
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::IsotropicVoceLawHardeningStressUpdateTempl(
    const InputParameters & parameters)
  : IsotropicLinearHardeningStressUpdateTempl<is_ad>(parameters),
    _q(this->template getParam<Real>("q")),
    _b(this->template getParam<Real>("b"))
{
}

template <bool is_ad>
GenericReal<is_ad>
IsotropicVoceLawHardeningStressUpdateTempl<is_ad>::computeHardeningValue(
    const GenericReal<is_ad> & scalar)
{
  _hardening_variable[_qp] = _q * (1.0 - std::exp(-_b * scalar));

  return (_hardening_variable_old[_qp] + _hardening_slope * scalar +
          _b * (_q - _hardening_variable_old[_qp]) *
              this->_effective_inelastic_strain_increment);
}

template class IsotropicVoceLawHardeningStressUpdateTempl<false>;
template class IsotropicVoceLawHardeningStressUpdateTempl<true>;
