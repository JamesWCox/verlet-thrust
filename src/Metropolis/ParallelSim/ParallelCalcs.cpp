#include <math.h>
#include "ParallelCalcs.h"
#include "Metropolis/GPUCopy.h"

Real ParallelCalcs::calcEnergy_LRC() {
    ParallelCalcs::LRC_Kernel<<<1,1>>>(GPUCopy::simBoxGPU());
    Real LRC = 0.0;
    cudaDeviceSynchronize();
    cudaMemcpy(&LRC, &(GPUCopy::simBoxGPU()->longRangeCorrection), sizeof(Real), cudaMemcpyDeviceToHost);
    return LRC;
}

__global__
void ParallelCalcs::LRC_Kernel(SimBox* sb) {
    ParallelCalcs::calculateLRC(sb);
}

__host__ __device__
void ParallelCalcs::calculateLRC(SimBox* sb) {

    int* molData = sb->moleculeData;
    Real* aData = sb->atomData;

	Real Ecut = 0.0;		// Holds LJ long-range cutoff energy correction

	// Volume of box in Ang^3
    Real Vnew = sb->size[X_COORD] * sb->size[Y_COORD] * sb->size[Z_COORD];
	Real RC3 = 1.00 / pow(sb->cutoff, 3);		// 1 / cutoff^3
	Real RC9 = pow(RC3, 3);						// 1 / cutoff^9

	// Note: currently only supports at most TWO solvents (needs to be updated for more)
	int a = 0, b = 1;
	Real NMOL1 = sb->numMolecules / 2;	// Number of molecules of solvent1
	Real NMOL2 = sb->numMolecules / 2;	// Number of molecules of solvent2

    int solvent1Start = molData[MOL_START * sb->numMolecules + a];
    int solvent1End = molData[MOL_LEN * sb->numMolecules + a] + solvent1Start;
	int NATOM1 = solvent1End - solvent1Start;			// Number of atoms in solvent1

    int solvent2Start = molData[MOL_START * sb->numMolecules + b];
    int solvent2End = molData[MOL_LEN * sb->numMolecules + b] + solvent2Start;
	int NATOM2 = solvent2End - solvent2Start;			// Number of atoms in solvent1

    int NATMX = NATOM1;
	if (NATMX < NATOM2) {		// NATMX = MAX(NAT0M1, NAT0M2)
		NATMX = NATOM2;
	}

	Real sig2, sig6, sig12;
	// get LJ-values for solvent1 and store in A6, A12
    Real* SigmaA = new Real[NATOM2];
    Real* EpsilonA = new Real[NATOM2];
    Real* A6 = new Real[NATOM2];
    Real* A12 = new Real[NATOM2];

	for(int i = 0; i < NATOM1; i++) {
        if(aData[ATOM_SIGMA * sb->numAtoms + solvent1Start + i] < 0.0
                || aData[ATOM_EPSILON * sb->numAtoms + solvent1Start + i] < 0.0) {
			SigmaA[i] = 0.0;
			EpsilonA[i] = 0.0;
		} else {
            SigmaA[i] = aData[ATOM_SIGMA * sb->numAtoms + solvent1Start + i];
            EpsilonA[i] = aData[ATOM_EPSILON * sb->numAtoms + solvent1Start + i];
		}

		sig2 = pow(SigmaA[i], 2);
        sig6 = pow(sig2, 3);
        sig12 = pow(sig6, 2);
		A6[i] = sqrt(4 * EpsilonA[i] * sig6);
		A12[i] = sqrt(4 * EpsilonA[i] * sig12);
	}

	// get LJ-values for solvent2 and store in B6, B12
    Real* SigmaB = new Real[NATOM2];
    Real* EpsilonB = new Real[NATOM2];
    Real* B6 = new Real[NATOM2];
    Real* B12 = new Real[NATOM2];

	for(int i = 0; i < NATOM2; i++) {
		if(aData[ATOM_SIGMA * sb->numAtoms + solvent2Start + i] < 0.0
                || aData[ATOM_EPSILON * sb->numAtoms + solvent2Start + i] < 0.0) {
			SigmaB[i] = 0.0;
			EpsilonB[i] = 0.0;
		} else {
            SigmaB[i] = aData[ATOM_SIGMA * sb->numAtoms + solvent2Start + i];
            EpsilonB[i] = aData[ATOM_EPSILON * sb->numAtoms + solvent2Start + i];
		}

		sig2 = pow(SigmaB[i], 2);
        sig6 = pow(sig2, 3);
        sig12 = pow(sig6, 2);
		B6[i] = sqrt(4 * EpsilonB[i] * sig6);
		B12[i] = sqrt(4 * EpsilonB[i] * sig12);
	}

	// loop over all atoms in a pair
	for(int i = 0; i < NATOM1; i++) {
		for(int j = 0; j < NATOM2; j++) {
			Ecut += (2*PI*NMOL1*NMOL1/(3.0*Vnew)) * (A12[i]*A12[j]*RC9/3.0 - A6[i]*A6[j]*RC3);
			Ecut += (2*PI*NMOL2*NMOL2/(3.0*Vnew)) * (B12[i]*B12[j]*RC9/3.0 - B6[i]*B6[j]*RC3);
			Ecut += (4*PI*NMOL1*NMOL2/(3.0*Vnew)) * (A12[i]*B12[j]*RC9/3.0 - A6[i]*B6[j]*RC3);
		}
	}
	sb->longRangeCorrection = Ecut;
}

