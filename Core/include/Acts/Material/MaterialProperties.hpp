// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// MaterialProperties.hpp, Acts project
///////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include "Acts/Material/Material.hpp"

namespace Acts {

/// @class MaterialProperties
///
/// Material with information associated to a thickness of material
/// This class is targeted for surface based material description.
/// A volume based material description is to be described by the
/// Material class.
///
/// the units are :
///  - thickness [mm] (only used for layer description)
///  - X0  [mm]
///  - L0  [mm]
///  - A   [g/mole]
///  - Z
///  - rho [g/mm3]
class MaterialProperties
{
public:
  /// Default Constructor
  MaterialProperties() = default;

  /// Constructor - for averaged material
  ///
  /// @param Xo is the radiation length in mm
  /// @param Lo is the nuclear interaction length in mm
  /// @param averageA is the average atomic weight
  /// @param averageZ is the average atomic number
  /// @param averageRho is the average density in g/mm3
  /// @param thickness is the thickness of the material
  MaterialProperties(float Xo,
                     float Lo,
                     float averageA,
                     float averageZ,
                     float averageRho,
                     float thickness);

  /// Constructor - for full Material class
  ///
  /// @param material is the material
  /// @param thickness is the thickness of the material
  MaterialProperties(const Material& material, float thickness);

  /// Constructor - for different layers of Material
  ///
  /// @param matLayers The vector of pairs of material and thickness
  /// @param unitThickness Boolean to set compound is set to unit thickness
  MaterialProperties(const std::vector<MaterialProperties>& matLayers,
                     bool unitThickness = true);

  /// Copy Constructor
  ///
  /// @param mprop is the source material properties to be copied
  MaterialProperties(const MaterialProperties& mprop) = default;

  /// Copy Move Constructor
  ///
  /// @param mprop is the source material properties to be copied
  MaterialProperties(MaterialProperties&& mprop) = default;

  /// Destructor
  virtual ~MaterialProperties() = default;

  /// Assignment Operator
  ///
  /// @param mprop is the source material properties object
  MaterialProperties&
  operator=(const MaterialProperties& mprop)
      = default;

  /// Assignment Move Operator
  ///
  /// @param mprop is the source material properties object
  MaterialProperties&
  operator=(MaterialProperties&& mprop)
      = default;

  /// Scale operator - scales the material thickness
  ///
  /// @param scale is the material scaling parameter
  MaterialProperties&
  operator*=(float scale);

  /// Comparison operator
  ///
  /// @param mprop is the source material properties object
  bool
  operator==(const MaterialProperties& mprop) const;

  /// Comparison operator
  ///
  /// @param mprop is the source material properties object
  bool
  operator!=(const MaterialProperties& mprop) const;

  /// Scale to unit thickness
  ///
  /// A helper method to allows to scale a material property
  /// for unphysical/blended material to a unit thickness of 1.
  /// This is safe for energy loss and multiple scattering
  /// application in the material integration
  ///
  /// Scaling to unit thickness changes:
  /// - X0,L0,rho of the material
  ///
  /// Leaves intact:
  /// - tInX0, tInL0, A, Z
  void
  scaleToUnitThickness();

  /// Average material properties
  ///
  /// This method creates an averaged material properties out of the new and
  /// the present material properties according to the following formulas:
  ///
  /// \f[
  ///	\frac{t}{x_0} = \sum_{i=1}^n \frac{t_i}{x_i}
  /// \f]
  /// \f[
  ///	\frac{t}{\Lambda_0} = \sum_{i=1}^n \frac{t_i}{\Lambda_i}
  /// \f]
  /// \f[
  ///	\rho = \frac{\sum_{i=1}^n t_i \rho_i}{\sum_{i=1}^n t_i}
  /// \f]
  /// \f[
  ///	A = \frac{\sum_{i=1}^n \rho_i A_i}{\sum_{i=1}^n \rho_i}
  /// \f]
  /// \f[
  ///	Z = \frac{\sum_{i=1}^n \rho_i Z_i}{\sum_{i=1}^n \rho_i}
  /// \f]
  /// t...thickness, \f$x_0\f$...radiation length, \f$\Lambda_0\f$...interaction
  /// length, \f$\rho\f$...density, A...mass number, Z...atomic number
  ///
  /// @param mprop are the material properties to be added in averaging
  void
  average(const MaterialProperties& mprop);

  /// Boolean operator
  /// false indicates it's vacuum
  operator bool() const { return bool(m_material); }

  /// Return the stored Material
  const Material&
  material() const;

  /// Return the thickness in mm
  float
  thickness() const;

  /// Return the radiationlength fraction
  float
  thicknessInX0() const;

  /// Return the nuclear interaction length fraction
  float
  thicknessInL0() const;

  /// Returns the average X0 of the material
  float
  averageX0() const;

  /// Return the average L0 of the material
  float
  averageL0() const;

  /// Returns the average Z of the material
  float
  averageZ() const;

  /// Return the average A of the material [gram/mole]
  float
  averageA() const;

  /// Return the average density of the material
  /// - in [g/mm^3]
  float
  averageRho() const;

  /// Return the @f$ Z/A * rho @f$
  float
  zOverAtimesRho() const;

protected:
  Material m_material = Material();  //!< the material
  float    m_dInX0{0.};              //!< thickness in units of radiation length
  float    m_dInL0{0.};  //!< thickness in units of nuclear interaction length
};

inline const Material&
MaterialProperties::material() const
{
  return m_material;
}

inline float
MaterialProperties::thicknessInX0() const
{
  return m_dInX0;
}

inline float
MaterialProperties::thicknessInL0() const
{
  return m_dInL0;
}

inline float
MaterialProperties::thickness() const
{
  return m_dInX0 * m_material.X0();
}

inline float
MaterialProperties::zOverAtimesRho() const
{
  return m_material.zOverAtimesRho();
}

inline float
MaterialProperties::averageX0() const
{
  return m_material.X0();
}

inline float
MaterialProperties::averageL0() const
{
  return m_material.L0();
}

inline float
MaterialProperties::averageA() const
{
  return m_material.A();
}

inline float
MaterialProperties::averageZ() const
{
  return m_material.Z();
}

inline float
MaterialProperties::averageRho() const
{
  return m_material.rho();
}

inline bool
MaterialProperties::operator==(const MaterialProperties& mp) const
{
  return (m_material == mp.m_material && m_dInX0 == mp.m_dInX0
          && m_dInL0 == mp.m_dInL0);
}

inline bool
MaterialProperties::operator!=(const MaterialProperties& mp) const
{
  return (!operator==(mp));
}

// Overload of << operator for std::ostream for debug output
std::ostream&
operator<<(std::ostream& sl, const MaterialProperties& mprop);

// Useful typedefs
using MaterialPropertiesVector = std::vector<MaterialProperties>;
using MaterialPropertiesMatrix = std::vector<MaterialPropertiesVector>;

}  // namespace
