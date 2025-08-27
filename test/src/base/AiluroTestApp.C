//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "AiluroTestApp.h"
#include "AiluroApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
AiluroTestApp::validParams()
{
  InputParameters params = AiluroApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

AiluroTestApp::AiluroTestApp(const InputParameters & parameters) : MooseApp(parameters)
{
  AiluroTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

AiluroTestApp::~AiluroTestApp() {}

void
AiluroTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  AiluroApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"AiluroTestApp"});
    Registry::registerActionsTo(af, {"AiluroTestApp"});
  }
}

void
AiluroTestApp::registerApps()
{
  registerApp(AiluroApp);
  registerApp(AiluroTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
AiluroTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  AiluroTestApp::registerAll(f, af, s);
}
extern "C" void
AiluroTestApp__registerApps()
{
  AiluroTestApp::registerApps();
}
