//#include "types.h"
#include "functionCPU.h"
#include "omp.h"
#include <algorithm>

using namespace NIP;

void WaveIteration(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b = 0.0)
{
    U(t % 3, y_s, x_s) += F(t - 1); // Source
    for (unsigned int i = 0; i < N * M; ++i) {
        int x = i % M;
        int y = i / M;
        if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
            U((t + 1) % 3, y, x) = 0.0; continue;
        }
        Real val;
        if (U(3, y, x) > 0.0) {
            val = -4.0 * U(t % 3, y, x);
            val += U(t % 3, y, x - 1);// (U(3, y, x - 1) > 0.0) ? U(t % 3, y, x - 1) : 0.0;//U[(x+1+y*M)*4+t%3];
            val += U(t % 3, y, x + 1);// (U(3, y, x + 1) > 0.0) ? U(t % 3, y, x + 1) : 0.0;//U[(x-1+y*M)*4+t%3];
            val += U(t % 3, y + 1, x);// (U(3, y + 1, x) > 0.0) ? U(t % 3, y + 1, x) : 0.0;//U[(x-M+y*M)*4+t%3];
            val += U(t % 3, y - 1, x);// (U(3, y - 1, x) > 0.0) ? U(t % 3, y - 1, x) : 0.0;//U[(x+M+y*M)*4+t%3];
            val *= U(3, y, x) * U(3, y, x) * v * v * d_t * d_t / (d_x * d_x); // ???
            val += 2.0 * U(t % 3, y, x) - U((t - 1) % 3, y, x) * (1.0 - d_t * b * 0.5);
            val /= (1.0 + d_t * b * 0.5);
        }
        else { val = 0.0; }
        U((t + 1) % 3, y, x) = val;
    }
    IR(t - 1) = U((t + 1) % 3, y_ir, x_ir); // (t-1) because we run with t+1
}

void WaveIterationOMP(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b, int Mth)
{
    if (Mth == -1)
        Mth = omp_get_max_threads();
    //omp_set_num_threads(Mth);

    U(t % 3, y_s, x_s) += F(t - 1); // Source
#pragma omp parallel for num_threads(Mth)
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            //int x = i % M;
            //int y = i / M;
            if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
                U((t + 1) % 3, y, x) = 0.0; continue;
            }
            Real val;
            if (U(3, y, x) > 0.0) {
                val = -4.0 * U(t % 3, y, x);
                val += U(t % 3, y, x - 1);// (U(3, y, x - 1) > 0.0) ? U(t % 3, y, x - 1) : 0.0;//U[(x+1+y*M)*4+t%3];
                val += U(t % 3, y, x + 1);// (U(3, y, x + 1) > 0.0) ? U(t % 3, y, x + 1) : 0.0;//U[(x-1+y*M)*4+t%3];
                val += U(t % 3, y + 1, x);// (U(3, y + 1, x) > 0.0) ? U(t % 3, y + 1, x) : 0.0;//U[(x-M+y*M)*4+t%3];
                val += U(t % 3, y - 1, x);// (U(3, y - 1, x) > 0.0) ? U(t % 3, y - 1, x) : 0.0;//U[(x+M+y*M)*4+t%3];
                val *= U(3, y, x) * U(3, y, x) * v * v * d_t * d_t / (d_x * d_x); // ???
                val += 2.0 * U(t % 3, y, x) - U((t - 1) % 3, y, x) * (1.0 - d_t * b * 0.5);
                val /= (1.0 + d_t * b * 0.5);
            }
            else { val = 0.0; }
            U((t + 1) % 3, y, x) = val;
        }
    }
    IR(t - 1) = U((t + 1) % 3, y_ir, x_ir); // (t-1) because we run with t+1
}

void WaveIterationOMPMultipleFrames(Array3D& U, Array1D& IR, const unsigned int N, const unsigned int M, const int t0, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Array1D& F, const Real b, const unsigned int iterations, int Mth)
{
    if (Mth == -1)
        Mth = omp_get_max_threads();

    for (int t = t0; t < t0 + iterations; ++t) {
        U(t % 3, y_s, x_s) += F(t - 1); // Source
#pragma omp parallel for num_threads(Mth)
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < M; ++x) {
                //int x = i % M;
                //int y = i / M;
                if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
                    U((t + 1) % 3, y, x) = 0.0; continue;
                }
                Real val;
                if (U(3, y, x) > 0.0) {
                    val = -4 * U(t % 3, y, x);
                    val += U(t % 3, y, x - 1);// (U(3, y, x - 1) > 0.0) ? U(t % 3, y, x - 1) : 0.0;//U[(x+1+y*M)*4+t%3];
                    val += U(t % 3, y, x + 1);// (U(3, y, x + 1) > 0.0) ? U(t % 3, y, x + 1) : 0.0;//U[(x-1+y*M)*4+t%3];
                    val += U(t % 3, y + 1, x);// (U(3, y + 1, x) > 0.0) ? U(t % 3, y + 1, x) : 0.0;//U[(x-M+y*M)*4+t%3];
                    val += U(t % 3, y - 1, x);// (U(3, y - 1, x) > 0.0) ? U(t % 3, y - 1, x) : 0.0;//U[(x+M+y*M)*4+t%3];
                    val *= U(3, y, x) * U(3, y, x) * v * v * d_t * d_t / (d_x * d_x); // ???
                    val += 2.0 * U(t % 3, y, x) - U((t - 1) % 3, y, x) * (1.0 - d_t * b * 0.5);
                    val /= (1.0 + d_t * b * 0.5);
                }
                else { val = 0.0; }
                U((t + 1) % 3, y, x) = val;
            }
        }
        IR(t - 1) = U((t + 1) % 3, y_ir, x_ir); // (t-1) because we run with t+1
    }
}

void WaveIterationAlt(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b)
{
    Real* walls = U + M * N * 3 * 2;
    Real* frame = U + M * N * 3 * (t % 2);
    Real* nextframe = U + M * N * 3 * ((t + 1) % 2);
    Real C = Real(0.5);
    frame[(x_s + y_s * M) * 3] += F[t - 1]; // Source
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
                nextframe[(x + y * M) * 3] = Real(0.0);
                nextframe[(x + y * M) * 3 + 1] = Real(0.0);;
                nextframe[(x + y * M) * 3 + 2] = Real(0.0);
                continue;
            }

            Real val1, val2, val3;
            Real Y, lY, bY;
            Real B, lB, bB;
            //Real R = Real(0.9);
            Real gradx, grady;
            Real vx, vy;
            Real lnf, bnf;

            B = walls[x + y * M];
            lB = walls[x - 1 + y * M];
            bB = walls[x - M + y * M];

            Y = (B > Real(0.0))? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
            lY = (lB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
            bY = (bB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);

            vx = frame[(x + y * M) * 3 + 1];
            vy = frame[(x + y * M) * 3 + 2];

            lnf = nextframe[(x - 1 + y * M) * 3];
            bnf = nextframe[(x - M + y * M) * 3];
           
            val1 = B * (frame[(x + y * M) * 3] - C * 
            //    ((frame[(x + y * M) * 3] - frame[(x - 1 + y * M) * 3]) * vx + (frame[(x + y * M) * 3] - frame[(x - M + y * M) * 3]) * vy));
                  ((frame[(x + 1 + y * M) * 3 + 1] - vx) + (frame[(x + M + y * M) * 3 + 2] - vy)));
            gradx = val1 - lnf;
            grady = val1 - bnf;
            //val2 = B * lB * (vx - C * gradx * val1)
            val2 = B * lB * (vx - C * gradx)
                + (lB - B) * (B * lY + lB * Y) * (B * val1 + lB * lnf);
            //val3 = B * bB * (vy - C * grady * val1)
            val3 = B * bB * (vy - C * grady)
                + (bB - B) * (B * bY + bB * Y) * (B * val1 + bB * bnf);

            nextframe[(x + y * M) * 3] = val1;
            nextframe[(x + y * M) * 3 + 1] = val2;
            nextframe[(x + y * M) * 3 + 2] = val3;
        }
    }
    IR[t - 1] = nextframe[(x_ir + y_ir * M) * 3];

}

void WaveIterationAltOMP(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b, int blocksize, int Mth)
{
    if (Mth == -1)
        Mth = omp_get_max_threads();

    Real* walls = U + M * N * 3 * 2;
    Real* frame = U + M * N * 3 * (t % 2);
    Real* nextframe = U + M * N * 3 * ((t + 1) % 2);
    Real C = Real(0.5);
    frame[(x_s + y_s * M) * 3] += F[t - 1]; // Source

    //int blocksize = 64;
    int Nblocks = N / blocksize + ((N % blocksize == 0) ? 0 : 1);
    int Mblocks = M / blocksize + ((M % blocksize == 0) ? 0 : 1);

    int Ndiag = Mblocks + Nblocks - 1;
    for (int diag = 0; diag < Ndiag; ++diag) {
        const int Eldiag = std::min({ diag + 1, Mblocks, Nblocks, Ndiag - diag });
#pragma omp parallel for num_threads(Mth) //firstprivate(diag,Eldiag,x1,y1,t1,t,fsize)
        for (int el = 0; el < Eldiag; ++el) {
            int xBlocks = std::min(diag, Mblocks - 1) - el;
            int yBlocks = el + ((diag >= Mblocks) ? (diag - Mblocks + 1) : 0);
            for (int y = yBlocks * blocksize; (y < (yBlocks + 1) * blocksize) && y < N; ++y) {
                for (int x = xBlocks * blocksize; (x < (xBlocks + 1) * blocksize) && x < M; ++x) {
                    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
                        nextframe[(x + y * M) * 3] = Real(0.0);
                        nextframe[(x + y * M) * 3 + 1] = Real(0.0);;
                        nextframe[(x + y * M) * 3 + 2] = Real(0.0);
                        continue;
                    }

                    Real val1, val2, val3;
                    Real Y, lY, bY;
                    Real B, lB, bB;
                    //Real R = Real(0.9);
                    Real gradx, grady;
                    Real vx, vy;
                    Real lnf, bnf;

                    B = walls[x + y * M];
                    lB = walls[x - 1 + y * M];
                    bB = walls[x - M + y * M];

                    Y = (B > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
                    lY = (lB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
                    bY = (bB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);

                    vx = frame[(x + y * M) * 3 + 1];
                    vy = frame[(x + y * M) * 3 + 2];

                    lnf = nextframe[(x - 1 + y * M) * 3];
                    bnf = nextframe[(x - M + y * M) * 3];

                    val1 = B * (frame[(x + y * M) * 3] - C *
                        ((frame[(x + 1 + y * M) * 3 + 1] - vx) + (frame[(x + M + y * M) * 3 + 2] - vy)));
                    gradx = val1 - lnf;
                    grady = val1 - bnf;
                    val2 = B * lB * (vx - C * gradx)
                        + (lB - B) * (B * lY + lB * Y) * (B * val1 + lB * lnf);
                    val3 = B * bB * (vy - C * grady)
                        + (bB - B) * (B * bY + bB * Y) * (B * val1 + bB * bnf);

                    nextframe[(x + y * M) * 3] = val1;
                    nextframe[(x + y * M) * 3 + 1] = val2;
                    nextframe[(x + y * M) * 3 + 2] = val3;
                }
            }
        }
    }
    IR[t - 1] = nextframe[(x_ir + y_ir * M) * 3];

}