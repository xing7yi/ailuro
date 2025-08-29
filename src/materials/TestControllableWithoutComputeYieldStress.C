#include "TestControllableWithoutComputeYieldStress.h"

registerMooseObject("AiluroApp", ADTestControllableWithoutComputeYieldStress);
registerMooseObject("AiluroApp", TestControllableWithoutComputeYieldStress);

template <bool is_ad>
InputParameters
TestControllableWithoutComputeYieldStressTempl<is_ad>::validParams()
{
  InputParameters params = CombinedNonlinearHardeningPlasticityTempl<is_ad>::validParams();
  
  params.addClassDescription("TEST: Combined plasticity model with controllable parameters "
                             "but WITHOUT overriding computeYieldStress - should NOT work properly");

  // 声明参数为controllable
  params.declareControllable("yield_stress isotropic_hardening_constant");

  return params;
}

template <bool is_ad>
TestControllableWithoutComputeYieldStressTempl<is_ad>::TestControllableWithoutComputeYieldStressTempl(
    const InputParameters & parameters)
  : CombinedNonlinearHardeningPlasticityTempl<is_ad>(parameters),
    _controllable_yield_stress(this->template getParam<Real>("yield_stress")),
    _controllable_isotropic_hardening_constant(this->template getParam<Real>("isotropic_hardening_constant"))
{
  // 添加调试输出
  this->_console << "TEST VERSION: Constructor called with yield_stress = " 
                 << _controllable_yield_stress << std::endl;
}

// 注意：我们故意不实现computeYieldStress，使用父类的版本

template class TestControllableWithoutComputeYieldStressTempl<false>;
template class TestControllableWithoutComputeYieldStressTempl<true>;
