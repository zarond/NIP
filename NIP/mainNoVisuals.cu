//#include <boost>
#include "types.h" 
//#include <chrono>
#include <fstream>
#include <iostream>
//#include <cmath>
#include "functionCPU.h"
#include "functionGPU.cuh"
#include "helper_kernels.cuh"
#include "CustomTimer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

//using namespace NIP;

//#define FREEGLUT_STATIC
//#define _LIB
//#define FREEGLUT_LIB_PRAGMAS 0 /////?????????????

#include <helper_gl.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_gl_interop.h>
//#include <vector_types.h>
//typedef float Real;

dim3 windowSize(512, 512);
dim3 windowBlockSize(32, 32, 1);
dim3 windowGridSize(windowSize.x / windowBlockSize.x, windowSize.y / windowBlockSize.y);

Real* d_Field = NULL;
Real* d_IR = NULL;
Real* d_F = NULL;
Real* d_FieldAlt = NULL;

int hz = 44100;// звук 44100 герц, 16 - битный
Real timestep = 1.0 / hz;//% ход симуляции для 44100 герц
Real v = 331;//% скорость звука м / с
Real d_x = v * timestep * 2;//% шаг сетки в пространстве 0.0075 м = 7.5 мм, но для симуляции надо брать больше X2
Real X_size = 10;//25;//% размеры комнаты
Real Y_size = 10;//25;
int N = int(Y_size / d_x);//% размеры в пикселях
int M = int(X_size / d_x);
//	int N = 1000;
//	int M = 1000;
int x_ir = -1;// int(M * 0.85);//% положение слушателя
int y_ir = -1;// int(N * 0.52);
int x_s = -1;//int(M * 0.5);//% положение источника
int y_s = -1;//int(N * 0.5);
int T = 100;
Real b = 0.5;
char filename[256] = "";
char F_filename[256] = "";
//Array3D Field(4, N, M);
//Array1D IR(2*T);
//Array1D F(2*T);
Array3D Field;
Array4D FieldAlt;
Array1D IR;
Array1D F;
int time_ind = 0;
const int NFrames = 3;
CustomTimer timer = CustomTimer();
int CalculationMode = 4;
int intensitySetting = 0;
Array3D WriteResults;
int JustRunSim = 0;
int RunSimMode = -1;
int writeresults = 0;
int Threads = 5;
const char* Methods[] = { "WaveIteration\t\t\t","WaveIterationOMP\t\t","WaveIterationOMPMultipleFrames\t","WaveIterationKernel\t\t", "MultipleIterationsKernel\t",
"WaveIterationKernelSM\t\t","WaveIterationAltOMP\t\t","WaveIterationKernelAlt\t\t" };

void saveIR();

void read_parameters(std::istream& in) {
	std::string name;
	if (!in == false)
	while (std::getline(in, name, '=')) {
		if (name.size() > 0 && name[0] == '#') in.ignore(1024 * 1024, '\n');
		else if (name == "hz") in >> hz;
		else if (name == "v") in >> v;
		else if (name == "X_size") in >> X_size;
		else if (name == "Y_size") in >> Y_size;
		else if (name == "x_ir") in >> x_ir;
		else if (name == "y_ir") in >> y_ir;
		else if (name == "x_s") in >> x_s;
		else if (name == "y_s") in >> y_s;
		else if (name == "T") in >> T;
		else if (name == "b") in >> b;
		else if (name == "room") in >> filename;
		else if (name == "JustRunSim") in >> JustRunSim;
		else if (name == "RunSimMode") in >> RunSimMode;
		else if (name == "F") in >> F_filename;
		else if (name == "WriteSimResults") in >> writeresults;
		else if (name == "Threads") in >> Threads;
		else {
			in.ignore(1024 * 1024, '\n');
			std::stringstream str;
			str << "Unknown parameter: " << name << '.';
			throw std::runtime_error(str.str().c_str());
		}
		in >> std::ws;
	}
	timestep = 1.0 / hz;
	d_x = v * timestep * 2;
	N = int(Y_size / d_x);
	M = int(X_size / d_x);
	if (x_ir < 0 || x_ir >= M) x_ir = int(M * 0.85);
	if (y_ir < 0 || y_ir >= N) y_ir = int(N * 0.52);
	if (x_s < 0 || x_s >= M) x_s = int(M * 0.5);
	if (y_s < 0 || y_s >= N) y_s = int(N * 0.5);
	//std::cout << filename << std::endl;
}

void saveIR() {
	if (d_IR != NULL && CalculationMode>=3) cudaMemcpy(IR.data(), d_IR, T * sizeof(Real), cudaMemcpyDeviceToHost);
	std::ofstream out("out.data");
	out << IR(blitz::Range(0,T-1));
	out.close();
}

bool loadF(char* name="in.data") {
	std::ifstream in(name);
	if (&in == nullptr) return false;
	in >> F;
	in.close();
	if (d_F == NULL) return true;
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	return true;
}

#define PI 3.14159265359

void loadDefaultF() {
	// Source
	float f0 = hz / 10;                    /* source dominant frequency, Hz */
	float t0 = 1.2 / f0;                /* source padding to move wavelet from left of zero */

	// Fill source waveform vecror
	float a = PI * PI * f0 * f0;            /* const for wavelet */
	float dt2dx2 = (timestep * timestep) / (d_x * d_x);   /* const for fd stencil */
	for (int it = 0; it < T; it++)
	{
		// Ricker wavelet (Mexican hat), second derivative of Gaussian
		//F(it) = 1e10 * (1.0 - 2.0 * a * pow(it * timestep - t0, 2)) * exp(-a * pow(it * timestep - t0, 2)) * dt2dx2;
		F(it) = 1e2 * exp(-a * .25 * pow(it * timestep - 4 / (PI * f0), 2));
	}
	if (d_F == NULL) return;
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	return;
}

void saveField() {
	std::ofstream out("outField.data");
	out << WriteResults;
	out.close();
}

void benchmark(int begN=1000,int endN=10000, int stepN=1000) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	N = endN;
	M = endN;
	x_ir = int(M * 0.85);
	y_ir = int(N * 0.52);
	x_s = int(M * 0.5);
	y_s = -int(N * 0.5);
	T = 200;
	b = 0.5;
	const unsigned int iter = 10;
	CustomTimer timer0 = CustomTimer();

	IR.resize(2 * T);
	F.resize(2 * T);
	IR = 0.0;
	F = 0.0;
	F(0) = 100.0;
	loadDefaultF();

	cudaMalloc((void**)&d_FieldAlt, 3 * 3 * M * N * sizeof(Real));

	cudaMalloc((void**)&d_Field, 4 * M * N * sizeof(Real));
	cudaMalloc((void**)&d_IR, T * sizeof(Real));
	cudaMalloc((void**)&d_F, T * sizeof(Real));
	cudaMemcpy(d_IR, IR.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	for (int sizes = begN; sizes <= endN; sizes += stepN) {
		N = sizes; M = sizes;
		x_ir = int(M * 0.85); y_ir = int(N * 0.52);
		x_s = int(M * 0.5); y_s = int(N * 0.5);
		Field.resize(4, N, M); /*Field = 0; */Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;

		FieldAlt.resize(3, N, M, 3);
		FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;

		windowSize.x = M;
		windowSize.y = N;
		windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
		for (CalculationMode=0; CalculationMode <= 7 ;++CalculationMode)
		{	
			Field(blitz::Range(0,2), blitz::Range::all(), blitz::Range::all()) = 0.0;
			cudaMemcpy(d_Field, Field.data(), 4 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
			//Field = 0; Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;
			if (CalculationMode == 6 || CalculationMode == 7) {
				FieldAlt(blitz::Range(0, 1), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 0.0;
				cudaMemcpy(d_FieldAlt, FieldAlt.data(), 3 * 2 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
			}
			for (time_ind = 0; time_ind < T;) {
				switch (CalculationMode) {
				case 0:
					timer0.tic();
					WaveIteration(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b); 
					timer0.toc(); break;
				case 1:
					timer0.tic();
					WaveIterationOMP(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b,Threads); 
					timer0.toc(); break;
				case 2:
					timer0.tic();
					WaveIterationOMPMultipleFrames(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b, iter, Threads);
					timer0.toc(iter);
					time_ind += iter - 1;
					break;
				case 3:
					timer0.tic();
					WaveIterationKernel << < windowGridSize, windowBlockSize >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
					cudaDeviceSynchronize(); 
					timer0.toc(); break;
				case 4:
					timer0.tic();
					MultipleIterationsKernel << < 1, 1 >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b, iter);
					cudaDeviceSynchronize();
					timer0.toc(iter);
					time_ind += iter - 1; break;
				case 5:
					timer0.tic();
					WaveIterationKernelSM << < windowGridSize, windowBlockSize, ((windowBlockSize.x + 2) * (windowBlockSize.y + 2) * 3) * sizeof(Real) >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
					cudaDeviceSynchronize();
					timer0.toc(); break;
				case 6:
					timer0.tic();
					WaveIterationAltOMP(FieldAlt.data(), IR.data(), N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F.data(), b, 64, Threads);
					timer0.toc();
					//cudaMemcpy(d_FieldAlt, FieldAlt.data(), 6 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
					//checkCudaErrors(cudaGetLastError());
					break;
				case 7:
					timer0.tic();
					WaveIterationKernelAlt << < 1, 1 >> > (d_FieldAlt, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, 32 ,b);
					cudaDeviceSynchronize();
					timer0.toc();
					//checkCudaErrors(cudaGetLastError());
					break;
				default:
					break;
				}		
				++time_ind;
			}
			if (CalculationMode >=3) checkCudaErrors(cudaGetLastError());
			TimerInfo timerinfo;
			if (CalculationMode == 4 || CalculationMode == 2)
				timerinfo = timer0.getInfo(min(timer0.N, T) / iter);
			else
				timerinfo = timer0.getInfo();
			//std::cout <<"size = "<< sizes << ", calculation mode = " << CalculationMode << ", mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion)
			//	<<" min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
			std::cout << "size = " << sizes << " \tMethod: " << Methods[CalculationMode]  << " mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion)
				<< " min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
		}
	}
	cudaFree(d_Field);
	cudaFree(d_FieldAlt);
	cudaFree(d_IR);
	cudaFree(d_F);

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "benchmark ended in " << diff << " seconds" << std::endl;
}

void benchmarkOMP(int begN = 1000, int endN = 10000, int stepN = 1000) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	N = endN;
	M = endN;
	x_ir = int(M * 0.85);
	y_ir = int(N * 0.52);
	x_s = int(M * 0.5);
	y_s = -int(N * 0.5);
	T = 200;
	b = 0.5;
	CustomTimer timer0 = CustomTimer();

	IR.resize(2 * T);
	F.resize(2 * T);
	IR = 0.0;
	F = 0.0;
	F(0) = 100.0;
	loadDefaultF();

	for (int sizes = begN; sizes <= endN; sizes += stepN) {
		N = sizes; M = sizes;
		x_ir = int(M * 0.85); y_ir = int(N * 0.52);
		x_s = int(M * 0.5); y_s = int(N * 0.5);
		Field.resize(4, N, M); /*Field = 0; */Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;
		windowSize.x = M;
		windowSize.y = N;
		windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
		for (int threads = 1; threads <= 16; ++threads)
		{
			Field(blitz::Range(0, 2), blitz::Range::all(), blitz::Range::all()) = 0.0;
			for (time_ind = 0; time_ind < T;) {
				timer0.tic();
				WaveIterationOMP(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b, threads);
				timer0.toc();
				++time_ind;
				//std::cout << time_ind << " ";
			}
			TimerInfo timerinfo;
			timerinfo = timer0.getInfo();
			std::cout << "size = " << sizes << " NumThreads = " << threads << " mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion) 
				<<" min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "benchmark ended in " << diff << " seconds" << std::endl;
}

void benchmarkOMPAlt(int begN = 1000, int endN = 10000, int stepN = 1000) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	N = endN;
	M = endN;
	x_ir = int(M * 0.85);
	y_ir = int(N * 0.52);
	x_s = int(M * 0.5);
	y_s = -int(N * 0.5);
	T = 200;
	b = 0.5;
	CustomTimer timer0 = CustomTimer();

	IR.resize(2 * T);
	F.resize(2 * T);
	IR = 0.0;
	F = 0.0;
	F(0) = 100.0;
	loadDefaultF();

	for (int sizes = begN; sizes <= endN; sizes += stepN) {
		N = sizes; M = sizes;
		x_ir = int(M * 0.85); y_ir = int(N * 0.52);
		x_s = int(M * 0.5); y_s = int(N * 0.5);

		FieldAlt.resize(3, N, M, 3);
		FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;

		windowSize.x = M;
		windowSize.y = N;
		windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
		for (int threads = 1; threads <= 16; ++threads)
		{
			FieldAlt(blitz::Range(0, 1), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 0.0;
			for (time_ind = 0; time_ind < T;) {
				timer0.tic();
				WaveIterationAltOMP(FieldAlt.data(), IR.data(), N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F.data(), b, 64, threads);
				timer0.toc();
				++time_ind;
				//std::cout << time_ind << " ";
			}
			TimerInfo timerinfo;
			timerinfo = timer0.getInfo();
			std::cout << "size = " << sizes << " NumThreads = " << threads << " mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion)
				<< " min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "benchmark ended in " << diff << " seconds" << std::endl;
}

void benchmarkKernel(int begN = 1000, int endN = 10000, int stepN = 1000) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	N = endN;
	M = endN;
	x_ir = int(M * 0.85);
	y_ir = int(N * 0.52);
	x_s = int(M * 0.5);
	y_s = -int(N * 0.5);
	T = 500;
	b = 0.5;
	const unsigned int iter = 10;
	CustomTimer timer0 = CustomTimer(400);

	IR.resize(2 * T);
	F.resize(2 * T);
	IR = 0.0;
	F = 0.0;
	F(0) = 100.0;
	loadDefaultF();

	cudaMalloc((void**)&d_Field, 4 * M * N * sizeof(Real));
	cudaMalloc((void**)&d_IR, T * sizeof(Real));
	cudaMalloc((void**)&d_F, T * sizeof(Real));
	cudaMemcpy(d_IR, IR.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	for (int sizes = begN; sizes <= endN; sizes += stepN) {
		N = sizes; M = sizes;
		x_ir = int(M * 0.85); y_ir = int(N * 0.52);
		x_s = int(M * 0.5); y_s = int(N * 0.5);
		Field.resize(4, N, M); /*Field = 0; */Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;
		windowSize.x = M;
		windowSize.y = N;
		windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
		for (int ksize = 1; ksize <= 32; ++ksize)
		{
			windowBlockSize.x = ksize;
			windowBlockSize.y = ksize;
			windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
			Field(blitz::Range(0, 2), blitz::Range::all(), blitz::Range::all()) = 0.0;
			//Field = 0; Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;
			cudaMemcpy(d_Field, Field.data(), 4 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
			for (time_ind = 0; time_ind < T;) {
				timer0.tic();
				WaveIterationKernel << < windowGridSize, windowBlockSize >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
				cudaDeviceSynchronize();
				timer0.toc();
				++time_ind;
			}
			checkCudaErrors(cudaGetLastError());
			TimerInfo timerinfo;
			timerinfo = timer0.getInfo();
			std::cout << "size = " << sizes << " blocksize = " << ksize << " mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion) 
				<< " min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
		}
	}
	cudaFree(d_Field);
	cudaFree(d_IR);
	cudaFree(d_F);

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "benchmark ended in " << diff << " seconds" << std::endl;
}

void benchmarkKernelAlt(int begN = 1000, int endN = 10000, int stepN = 1000) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	N = endN;
	M = endN;
	x_ir = int(M * 0.85);
	y_ir = int(N * 0.52);
	x_s = int(M * 0.5);
	y_s = -int(N * 0.5);
	T = 500;
	b = 0.5;
	const unsigned int iter = 10;
	CustomTimer timer0 = CustomTimer(400);

	IR.resize(2 * T);
	F.resize(2 * T);
	IR = 0.0;
	F = 0.0;
	F(0) = 100.0;
	loadDefaultF();

	cudaMalloc((void**)&d_FieldAlt, 3 * 3 * M * N * sizeof(Real));
	cudaMalloc((void**)&d_IR, T * sizeof(Real));
	cudaMalloc((void**)&d_F, T * sizeof(Real));
	cudaMemcpy(d_IR, IR.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	for (int sizes = begN; sizes <= endN; sizes += stepN) {
		N = sizes; M = sizes;
		x_ir = int(M * 0.85); y_ir = int(N * 0.52);
		x_s = int(M * 0.5); y_s = int(N * 0.5);
		FieldAlt.resize(3, N, M, 3);
		FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;
		windowSize.x = M;
		windowSize.y = N;
		windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
		for (int ksize = 1; ksize <= 32; ++ksize)
		{
			windowBlockSize.x = ksize;
			windowBlockSize.y = ksize;
			windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x == 0) ? 0 : 1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
			FieldAlt(blitz::Range(0, 1), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 0.0;
			cudaMemcpy(d_FieldAlt, FieldAlt.data(), 3 * 2 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
			for (time_ind = 0; time_ind < T;) {
				timer0.tic();
				WaveIterationKernelAlt << < 1, 1 >> > (d_FieldAlt, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, ksize, b);
				cudaDeviceSynchronize();
				timer0.toc();
				++time_ind;
			}
			checkCudaErrors(cudaGetLastError());
			TimerInfo timerinfo;
			timerinfo = timer0.getInfo();
			std::cout << "size = " << sizes << " blocksize = " << ksize << " mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion) 
				<< " min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
		}
	}
	cudaFree(d_FieldAlt);
	cudaFree(d_IR);
	cudaFree(d_F);

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "benchmark ended in " << diff << " seconds" << std::endl;
}

void RunAndSaveResults(int mode = 0, int writeresults=0) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	CustomTimer timer0 = CustomTimer();
	if (writeresults) {
		try { WriteResults.resize(T, N, M); }
		catch (...) { std::cout << "memory error, array too big"; return; }
	}
	//if (mode == 1) CalculationMode = 1;
	//if (mode == 1) CalculationMode = 2;
	//if (mode == 2) CalculationMode = 3;
	//if (mode == 3) CalculationMode = 4;
	//if (mode == 4) CalculationMode = 5;
	//if (mode == 5) CalculationMode = 6;

	int iter = T;
	for (time_ind = 0; time_ind < T;) {
		switch (mode) {
		case 0:
			timer0.tic();
			WaveIteration(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b);
			timer0.toc();
			if (writeresults) WriteResults(time_ind % T, blitz::Range::all(), blitz::Range::all()) = Field((time_ind + 2) % 3, blitz::Range::all(), blitz::Range::all());
			break;
		case 1:
			timer0.tic();
			WaveIterationOMP(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b, Threads);
			timer0.toc();
			if (writeresults) WriteResults(time_ind % T, blitz::Range::all(), blitz::Range::all()) = Field((time_ind + 2) % 3, blitz::Range::all(), blitz::Range::all()); 
			break;
		case 2:
			timer0.tic();
			WaveIterationOMPMultipleFrames(Field, IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F, b, iter, Threads);
			timer0.toc(iter);
			time_ind += iter - 1;
			break;
		case 3:
			timer0.tic();
			WaveIterationKernel <<< windowGridSize, windowBlockSize >>>  (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
			cudaDeviceSynchronize();
			timer0.toc();
			if (writeresults) cudaMemcpy(WriteResults.data() + M * N * (time_ind % T), d_Field + ((time_ind + 2) % 3) * M * N, M * N * sizeof(Real), cudaMemcpyDeviceToHost); 
			break;
		case 4:
			timer0.tic();
			MultipleIterationsKernel <<< 1, 1 >>> (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b, iter);
			cudaDeviceSynchronize();
			time_ind += iter - 1;
			timer0.toc(iter);
			checkCudaErrors(cudaGetLastError());
			break;
		case 5:
			timer0.tic();
			WaveIterationKernelSM << < windowGridSize, windowBlockSize, ((windowBlockSize.x + 2) * (windowBlockSize.y + 2) * 3) * sizeof(Real) >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
			//WaveIterationKernelSM << < windowGridSize, windowBlockSize >> > (d_Field, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, b);
			cudaDeviceSynchronize();
			timer0.toc();
			if (writeresults) cudaMemcpy(WriteResults.data() + M * N * (time_ind % T), d_Field + ((time_ind + 2) % 3) * M * N, M * N * sizeof(Real), cudaMemcpyDeviceToHost);
			break;
		case 6:
			timer0.tic();
			WaveIterationAltOMP(FieldAlt.data(), IR.data(), N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, F.data(), b, 64, Threads);
			timer0.toc();
			//cudaMemcpy(d_FieldAlt, FieldAlt.data(), 6 * M * N * sizeof(Real), cudaMemcpyHostToDevice);
			//if (writeresults) cudaMemcpy(WriteResults.data() + M * N * (time_ind % T), d_Field + ((time_ind + 2) % 3) * M * N, M * N * sizeof(Real), cudaMemcpyDeviceToHost);
			break;
		case 7:
			timer0.tic();
			WaveIterationKernelAlt << < 1, 1 >> > (d_FieldAlt, d_IR, N, M, time_ind % T + 1, v, d_x, timestep, x_ir, y_ir, x_s, y_s, d_F, 32, b);
			cudaDeviceSynchronize();
			timer0.toc();
			//checkCudaErrors(cudaGetLastError());
			break;
		default:
			break;
		}
		++time_ind;
	}
	if (mode >= 2) { 
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(IR.data(), d_IR, T * sizeof(Real), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError()); 
	}
	TimerInfo timerinfo;
	//timerinfo = timer0.getInfo();
	if (mode == 4 || mode == 2) timerinfo = timer0.getInfo(1);
	else timerinfo = timer0.getInfo();
	std::cout <<"runSim method = " << Methods[mode] << ", mean = " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion)
		<<" min,max= " << timerinfo.min << " " << timerinfo.max << std::endl;
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "running ended in " << diff << " milliseconds" << std::endl;
	start = std::chrono::high_resolution_clock::now();
	saveIR();
	if (writeresults) {
		saveField();
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
		std::cout << "writing ended in " << diff << " seconds" << std::endl;
	}

	cudaFree(d_Field);
	cudaFree(d_IR);
	cudaFree(d_F);
	cudaFree(d_FieldAlt);
}

int main(int argc, char** argv) {
	using namespace NIP;

	int myNr = 0;
	if (argc > 1)
		myNr = std::stoi(argv[1]);
	CalculationMode = myNr;

	if (CalculationMode == -1) { benchmark(500,5000, 500); return 0; }
	if (CalculationMode == -2) { benchmarkOMP(500, 5000, 500); return 0; }
	if (CalculationMode == -3) { benchmarkOMPAlt(500, 5000, 500); return 0; }
	if (CalculationMode == -4) { benchmarkKernel(500, 5000, 500); return 0; }
	if (CalculationMode == -5) { benchmarkKernelAlt(500, 5000, 500); return 0; }

	const char* input_filename = (argc > 2) ? argv[2] : "config.model";
	std::ifstream cfg(input_filename);
	read_parameters(cfg);
	cfg.close();

	//if (CalculationMode == 5 || JustRunSim && RunSimMode == 4) NFrames = 3; else NFrames = 3;

	if (filename[0] != '\0') {
		std::ifstream room(filename,std::ios_base::binary);
		int x, y, n;
		unsigned char *data = stbi_load(filename, &x, &y, &n, 1);
		if (data != nullptr) {
			N = y;
			M = x;
			X_size = M * d_x;
			Y_size = N * d_x;
			if (x_ir < 0 || x_ir >= M) x_ir = int(M * 0.85);
			if (y_ir < 0 || y_ir >= N) y_ir = int(N * 0.52);
			if (x_s < 0 || x_s >= M) x_s = int(M * 0.5);
			if (y_s < 0 || y_s >= N) y_s = int(N * 0.5);
			Field.resize(NFrames+1, N, M);
			Field = 0;
			blitz::Array<unsigned char, 2> tmp(data, blitz::shape(N, M), blitz::duplicateData);

			FieldAlt.resize(3, N, M, 3);
			FieldAlt = 0.0;
			for (int y0 = 0; y0 < N; ++y0) {
				for (int x0 = 0; x0 < M; ++x0) {
					Field(NFrames, y0, x0) = ((Real)tmp(y0, x0)) / 255.0;
					*(FieldAlt.data() + 3*2*M*N +x0+M*y0) = ((Real)tmp(y0, x0)) / 255.0;
				}
			}
			//tmp.free();
			//std::cout << Field(3,0,0) << " " <<Field(3, 100, 100);
			stbi_image_free(data);	
		}
		else { Field.resize(NFrames+1, N, M); Field = 0; Field(NFrames, blitz::Range::all(), blitz::Range::all()) = 1.0;
			FieldAlt.resize(3,N,M,3);
			FieldAlt = 0.0;
			FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;}
		room.close();
	}
	else { Field.resize(NFrames+1, N, M); Field = 0; Field(NFrames, blitz::Range::all(), blitz::Range::all()) = 1.0; 
		FieldAlt.resize(3, N, M, 3);
		FieldAlt = 0.0;
		FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;
	}

	IR.resize(2 * T);
	F.resize(2 * T);

	//Field = 0;
	//Field(3, blitz::Range::all(), blitz::Range::all()) = 1.0;
	IR = 0;
	if (F_filename[0] != '\0') {
		bool b = loadF(F_filename);
		if (b == false) {
			F = 0.0;
			F(0) = 100.0;
			loadDefaultF();
		}
	}
	else {
		F = 0.0;
		F(0) = 100.0;
		loadDefaultF();
	}

	//FieldAlt.resize(3,N,M,3);
	//FieldAlt = 0.0;
	//FieldAlt(2, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()) = 1.0;
	cudaMalloc((void**)&d_FieldAlt,  3 * 3 * M * N * sizeof(Real));
	cudaMemcpy(d_FieldAlt, FieldAlt.data(), 3 * 3 * M * N * sizeof(Real), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_Field, (NFrames + 1) * M * N * sizeof(Real));
	cudaMemcpy(d_Field, Field.data(), (NFrames + 1) * M * N * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_IR, T * sizeof(Real));
	cudaMemcpy(d_IR, IR.data(), T * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_F, T * sizeof(Real));
	cudaMemcpy(d_F, F.data(), T * sizeof(Real), cudaMemcpyHostToDevice);


	windowSize.x = M;
	windowSize.y = N;
	windowGridSize = dim3(windowSize.x / windowBlockSize.x + ((windowSize.x % windowBlockSize.x==0)? 0:1), windowSize.y / windowBlockSize.y + ((windowSize.y % windowBlockSize.y == 0) ? 0 : 1));
	//windowGridSize = dim3(windowSize.x / windowBlockSize.x, windowSize.y / windowBlockSize.y);
	
	if (JustRunSim) {
		RunSimMode = (RunSimMode >= 0)? RunSimMode : CalculationMode;
		RunAndSaveResults(RunSimMode,writeresults);
		return 0;
	}
	std::cout << "Method: " << Methods[CalculationMode] << std::endl;


	cudaFree(d_Field);
	cudaFree(d_IR);
	cudaFree(d_F);
	return 0;
}