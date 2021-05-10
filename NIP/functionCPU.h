#include "types.h"

using namespace NIP;

void WaveIteration(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b);
void WaveIterationOMP(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b,int Mth = -1);
void WaveIterationOMPMultipleFrames(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t0, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b, const unsigned int iterations = 1, int Mth = -1);
void WaveIterationAlt(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b = Real(0.0));
void WaveIterationAltOMP(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b = Real(0.0), int blocksize = 64, int Mth=-1);
