#include "BruteForceStep.h"
#include "ProximityMatrixStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"

#ifdef _OPENACC
#include <openacc.h>
#endif


ProximityMatrixStep::~ProximityMatrixStep() {
  ProximityMatrixCalcs::freeProximityMatrix(this->proximityMatrix);
  this->proximityMatrix = NULL;
}

Real ProximityMatrixStep::calcMolecularEnergyContribution(int currMol,
                                                          int startMol) {
  return ProximityMatrixCalcs::calcMolecularEnergyContribution( currMol,
      startMol, this->proximityMatrix);
}

Real ProximityMatrixStep::calcSystemEnergy(Real &subLJ, Real &subCharge,
                                           int numMolecules) {
  Real result = SimulationStep::calcSystemEnergy(subLJ, subCharge,
                                                 numMolecules);
  this->proximityMatrix = ProximityMatrixCalcs::createProximityMatrix();
  return result;
}

void ProximityMatrixStep::changeMolecule(int molIdx, SimBox *box) {
  SimulationStep::changeMolecule(molIdx, box);
  ProximityMatrixCalcs::updateProximityMatrix(this->proximityMatrix, molIdx);
}

void ProximityMatrixStep::rollback(int molIdx, SimBox *box) {
  SimulationStep::rollback(molIdx, box);
  ProximityMatrixCalcs::updateProximityMatrix(this->proximityMatrix, molIdx);
}

// ----- ProximityMatrixCalcs Definitions -----

Real ProximityMatrixCalcs::calcMolecularEnergyContribution(
    int currMol, int startMol, char *proximityMatrix) {
  Real total = 0;

  SimBox* sb = GPUCopy::simBoxCPU();
  int *molData = sb->moleculeData;
  Real *atomCoords = sb->atomCoordinates;
  Real *bSize = sb->size;
  int *pIdxes = sb->primaryIndexes;
  Real *aData = sb->atomData;
  Real cutoff = sb->cutoff;
  const long numMolecules = sb->numMolecules;
  const int numAtoms = sb->numAtoms;

  const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
  const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

  if (proximityMatrix == NULL) {
    for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
      if (otherMol != currMol) {
        int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
        int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
        if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                       atomCoords, bSize, pIdxes, cutoff, numAtoms)) {
          total += calcMoleculeInteractionEnergy(currMol, otherMol, molData,
                                                 aData, atomCoords, bSize,
                                                 numMolecules, numAtoms);
        }
      }
    }
  } else {
    for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
      if (otherMol != currMol) {
        if (proximityMatrix[currMol*numMolecules + otherMol]) {
          total += calcMoleculeInteractionEnergy(currMol, otherMol, molData,
                                                 aData, atomCoords, bSize,
                                                 numMolecules, numAtoms);
        }
      }
    }
  }

  return total;
}

// TODO: Duplicate; abstract out when PGCC supports it
Real ProximityMatrixCalcs::calcMoleculeInteractionEnergy (int m1, int m2,
                                                          int* molData,
                                                          Real* aData,
                                                          Real* aCoords,
                                                          Real* bSize,
                                                          int numMolecules,
                                                          int numAtoms) {
  Real energySum = 0;

  const int m1Start = molData[MOL_START * numMolecules + m1];
  const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

  const int m2Start = molData[MOL_START * numMolecules + m2];
  const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

  for (int i = m1Start; i < m1End; i++) {
    for (int j = m2Start; j < m2End; j++) {
      if (aData[ATOM_SIGMA * numAtoms + i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
          && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

        const Real r2 = SimCalcs::calcAtomDistSquared(i, j, aCoords, bSize, numAtoms);
        if (r2 == 0.0) {
          energySum += 0.0;
        } else {
          energySum += SimCalcs::calcLJEnergy(i, j, r2, aData, numAtoms);
          energySum += SimCalcs::calcChargeEnergy(i, j, sqrt(r2), aData, numAtoms);
        }
      }
    }
  }

  return (energySum);
}

char *ProximityMatrixCalcs::createProximityMatrix() {

  SimBox* sb = GPUCopy::simBoxCPU();
  const long numMolecules = sb->numMolecules;
  const Real cutoff = sb->cutoff;

  int* molData = sb->moleculeData;
  Real* atomCoords = sb->atomCoordinates;
  Real* bSize = sb->size;
  int* pIdxes = sb->primaryIndexes;
  int numAtoms = sb->numAtoms;

  char *matrix;
  #ifdef _OPENACC
  if (SimCalcs::on_gpu) {
    matrix = (char *)acc_malloc(numMolecules * numMolecules * sizeof(char));
  } else {
    matrix = (char *)malloc(numMolecules * numMolecules * sizeof(char));
  }
  #else
  matrix = (char *)malloc(numMolecules * numMolecules * sizeof(char));
  #endif
  assert(matrix != NULL);

  for (int i = 0; i < numMolecules; i++) {
    const int p1Start = molData[MOL_PIDX_START * numMolecules + i];
    const int p1End   = molData[MOL_PIDX_COUNT * numMolecules + i] + p1Start;
    for (int j = 0; j < numMolecules; j++) {
      const int p2Start = molData[MOL_PIDX_START * numMolecules + j];
      const int p2End = molData[MOL_PIDX_COUNT * numMolecules + j] + p2Start;
      matrix[i*numMolecules + j] = (j != i &&
          SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                     atomCoords, bSize, pIdxes, cutoff, numAtoms));
    }
  }
  return matrix;
}

void ProximityMatrixCalcs::updateProximityMatrix(char *matrix, int i) {
  SimBox* sb = GPUCopy::simBoxCPU();
  const long numMolecules = sb->numMolecules;
  const Real cutoff = sb->cutoff;

  int* molData = sb->moleculeData;
  Real* atomCoords = sb->atomCoordinates;
  Real* bSize = sb->size;
  int* pIdxes = sb->primaryIndexes;
  int numAtoms = sb->numAtoms;

  for (int j = 0; j < numMolecules; j++) {
    const int p1Start = molData[MOL_PIDX_START * numMolecules + i];
    const int p1End   = molData[MOL_PIDX_COUNT * numMolecules + i] + p1Start;
    const int p2Start = molData[MOL_PIDX_START * numMolecules + j];
    const int p2End   = molData[MOL_PIDX_COUNT * numMolecules + j] + p2Start;

    const char entry = (j != i && SimCalcs::moleculesInRange(p1Start, p1End,
                                                             p2Start, p2End,
                                                             atomCoords, bSize,
                                                             pIdxes, cutoff, numAtoms));
    matrix[i*numMolecules + j] = entry;
    matrix[j*numMolecules + i] = entry;
  }
}

void ProximityMatrixCalcs::freeProximityMatrix(char *matrix) {
  #ifdef _OPENACC
  if (SimCalcs::on_gpu)
    acc_free(matrix);
  else
  #endif
    free(matrix);
}
