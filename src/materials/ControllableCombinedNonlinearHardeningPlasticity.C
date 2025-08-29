#include "ControllableCombinedNonlinearHardeningPlasticity.h"
#include "Function.h"

registerMooseObject("AiluroApp", ADControllableCombinedNonlinearHardeningPlasticity);
registerMooseObject("AiluroApp", ControllableCombinedNonlinearHardeningPlasticity);

template <bool is_ad>
InputParameters
ControllableCombinedNonlinearHardeningPlasticityTempl<is_ad>::validParams()
{
  InputParameters params = CombinedNonlinearHardeningPlasticityTempl<is_ad>::validParams();
  
  params.addClassDescription("Combined isotropic and kinematic plasticity model with controllable "
                             "yield_stress and isotropic_hardening_constant parameters for "
                             "StochasticTools module compatibility.");

  // Override the parent parameters to make them controllable
  params.declareControllable("yield_stress isotropic_hardening_constant");

  return params;
}

template <bool is_ad>
ControllableCombinedNonlinearHardeningPlasticityTempl<is_ad>::ControllableCombinedNonlinearHardeningPlasticityTempl(
    const InputParameters & parameters)
  : CombinedNonlinearHardeningPlasticityTempl<is_ad>(parameters),
    _controllable_yield_stress(this->template getParam<Real>("yield_stress")),
    _controllable_isotropic_hardening_constant(this->template getParam<Real>("isotropic_hardening_constant"))
{
  // 添加调试输出
  this->_console << "CORRECT VERSION: Constructor called with yield_stress = " 
                 << _controllable_yield_stress << std::endl;
}

template <bool is_ad>
void
ControllableCombinedNonlinearHardeningPlasticityTempl<is_ad>::computeYieldStress(
    const GenericRankFourTensor<is_ad> & elasticity_tensor)
{
  // Handle yield_stress_function case
  if (this->_yield_stress_function)
  {
    static const Moose::GenericType<Point, is_ad> p;
    GenericReal<is_ad> function_yield_stress = this->_yield_stress_function->value(this->_temperature[this->_qp], p);
    if (function_yield_stress <= 0.0)
      mooseError("In ",
                 this->_name,
                 ": The calculated yield stress (",
                 function_yield_stress,
                 ") is less than zero");
    // 直接设置父类的yield stress为函数计算值
    const_cast<GenericReal<is_ad>&>(this->_yield_stress) = function_yield_stress;
  }
  else
  {
    // 直接使用controllable参数设置父类的yield stress
    const_cast<GenericReal<is_ad>&>(this->_yield_stress) = _controllable_yield_stress;
  }
  
  // 同样更新isotropic_hardening_constant以确保controllable生效
  const_cast<Real&>(this->_isotropic_hardening_constant) = _controllable_isotropic_hardening_constant;
  
  this->_console << "CORRECT VERSION: Set parent _yield_stress to " 
                 << this->_yield_stress 
                 << ", isotropic_hardening_constant to " 
                 << this->_isotropic_hardening_constant << std::endl;
}


template class ControllableCombinedNonlinearHardeningPlasticityTempl<false>;
template class ControllableCombinedNonlinearHardeningPlasticityTempl<true>;
