#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <windows.h>


using namespace std;

void showMatriz(int *matriz, int anchura, int altura);
void generateSeeds(int *matriz, int ancho, int alto, int cantidad, char modo);
void gestionSemillas(int *matriz, int ancho, int numeroSemillas, int alto, char modo);
int checkFull(int *matriz, int tamano);
bool checkMove(int *matriz, int ancho, int alto);
void guardar(int vidas, int *tablero, int altura, int anchura, char dificultad);
int* cargar();
int* MostrarEspecificaciones();

cudaError_t cudaStatus;

/*	add_up
*	Función del kernel para sumar hacia arriba todos los números que sean iguales.
*/
__device__ void add_up(int *matriz, int x, int y, int altura, int anchura)
{
	if (x != 0 && y < anchura) //Los primeros hilos no deben realizar ninguna operacion pues serán modificados por los demas
	{
		if (matriz[x*anchura + y] != 0) //Si es distinto de 0, gestiona su posible suma o desplazamiento
		{
			if (matriz[x*anchura + y] == matriz[(x - 1)*anchura + y]) //Si es igual a su superior, se procede a comprobar el numero de celdas con el mismo numero que hay en esa columna
			{
				int iguales = 0;
				iguales++;
				for (int i = 1; i <= x; i++)
				{
					if (matriz[x*anchura + y] == matriz[(x - i)*anchura + y])
					{
						iguales++;
					}
					else {
						break;
					}
				}
				if (iguales % 2 == 0) //Si el numero es par, se suman, si no, ese numero será mezclado con otro y no estará disponible
				{
					matriz[(x - 1)*anchura + y] = matriz[(x - 1)*anchura + y] * 2;
					matriz[x*anchura + y] = 0;
				}
			}
			else if (matriz[(x - 1)*anchura + y] == 0) //Se comprueba que otros hilos hayan dejado 0 en sus operaciones para desplazarse
			{
				matriz[(x - 1)*anchura + y] = matriz[x*anchura + y];
				matriz[x*anchura + y] = 0;
			}
		}
	}
}

/*	stack_up
*	Función del kernel para desplazar todos los números hacia arriba.
*/
__device__ void stack_up(int *matriz, int anchura, int altura, int x, int y) {

	for (int i = altura - 1; i > 0; i--) //realizaremos el desplazamiento celda a celda una altura-1 veces para gestionar la posibilidad del ultimo poniendose el primero de la lista
	{
		if ((x != 0) && (matriz[x*anchura + y] != 0) && matriz[x*anchura + (y - anchura)] == 0) //Si la celda pertenece a la primera fila, es 0 o su superior no es 0, no hace nada
		{
			matriz[x*anchura + (y - anchura)] = matriz[x*anchura + y];	//Si lo es, desplazamos la celda
			matriz[x*anchura + y] = 0;
		}
		__syncthreads();	//utilizamos una sincronizacion para que estos pasos sean realizados a la vez por los hilos del bloque y 
	}
}
/*	mov_upK
*	Kernel que gestiona las operaciones para mover hacia arriba los numeros, sumandolos en el proceso
*/
__global__ void mov_upK(int *matriz, int anchura, int altura) {
	int x = blockIdx.x;
	int y = threadIdx.y;

	stack_up(matriz, anchura, altura, x, y);	//Realizamos las llamadas de la siguiente manera para gestionar el movimiento:
	add_up(matriz, x, y, altura, anchura);		//2 2 0 4   -> 4 4 0 0
	__syncthreads();
	stack_up(matriz, anchura, altura, x, y);
}

/*	move_up
*	Metodo que gestiona la llamada al kerner mov_upK
*/
cudaError_t move_up(int *matriz, int ancho, int alto) {
	cudaError_t cudaStatus;

	int *dev_m;		//Establecemos la matriz donde se van a recoger los resultados


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en setdevice");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_m, ancho*alto * sizeof(int));	//Reservamos memoria para la matriz resultado
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en Malloc");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_m, matriz, ancho*alto * sizeof(int), cudaMemcpyHostToDevice); //copiamos los datos iniciales
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	//Establecemos las dimensiones pertinentes
	dim3 dimgrid(alto, 1);
	dim3 dimblock(1, ancho, 1);

	mov_upK << < dimgrid, dimblock >> > (dev_m, ancho, alto);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error en synchronize mov_up\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en mov_up");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matriz, dev_m, ancho*alto * sizeof(int), cudaMemcpyDeviceToHost); //recogemos el resultado en la variable del host
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en memcpy to host de mov_upK");
		goto Error;
	}

Error:	//Liberamos la memoria de la variable en caso de error
	cudaFree(dev_m);

	return cudaStatus;
}

/*	add_down
*	Función del kernel para sumar hacia la abajo todos los números que sean iguales.
*/
__device__ void add_down(int *matriz, int x, int y, int altura, int anchura)
{
	if (x != altura - 1 && y < anchura) //Los ultimos hilos no deben realizar ninguna operacion pues serán modificados por los demas
	{
		if (matriz[x*anchura + y] != 0) //Si es distinto de 0, gestiona su posible suma o desplazamiento
		{
			if (matriz[x*anchura + y] == matriz[(x + 1)*anchura + y]) //Si es igual a su inferior, se procede a comprobar el numero de celdas con el mismo numero que hay en esa columna
			{
				int iguales = 0;
				iguales++;
				for (int i = 1; x + i <= altura; i++)
				{
					if (matriz[x*anchura + y] == matriz[(x + i)*anchura + y])
					{
						iguales++;
					}
					else {
						break;
					}
				}
				if (iguales % 2 == 0) //Si el numero es par, se suman, si no, ese numero será mezclado con otro y no estará disponible
				{
					matriz[(x + 1)*anchura + y] = matriz[(x + 1)*anchura + y] * 2;
					matriz[x*anchura + y] = 0;
				}
			}
			else if (matriz[(x + 1)*anchura + y] == 0) //Se comprueba que otros hilos hayan dejado 0 en sus operaciones para desplazarse
			{
				matriz[(x + 1)*anchura + y] = matriz[x*anchura + y];
				matriz[x*anchura + y] = 0;
			}
		}
	}
}

/*	stack_down
*	Función del kernel para desplazar todos los números hacia abajo.
*/
__device__ void stack_down(int *matriz, int anchura, int altura, int x, int y) {

	for (int i = altura - 1; i > 0; i--)  //realizaremos el desplazamiento celda a celda una altura-1 veces para gestionar la posibilidad del ultimo poniendose el primero de la lista
	{
		if ((x != altura - 1) && (matriz[x*anchura + y] != 0) && matriz[(x + 1)*anchura + y] == 0) //Si la celda pertenece a la primera fila, es 0 o su superior no es 0, no hace nada
		{
			matriz[(x + 1)*anchura + y] = matriz[x*anchura + y]; //Si lo es, desplazamos la celda
			matriz[x*anchura + y] = 0;
		}
		__syncthreads();
	}
}


/*	mov_downK
*	Kernel que gestiona las operaciones para mover hacia abajo los numeros, sumandolos en el proceso
*/
__global__ void mov_downK(int *matriz, int anchura, int altura) {
	int x = blockIdx.x;
	int y = threadIdx.y;

	stack_down(matriz, anchura, altura, x, y);		//Realizamos las llamadas de la siguiente manera para gestionar el movimiento:
	add_down(matriz, x, y, altura, anchura);		//2 2 0 4   -> 4 4 0 0
	__syncthreads();
	stack_down(matriz, anchura, altura, x, y);
}

/*	move_down
*	Metodo que gestiona la llamada al kerner mov_downK
*/
cudaError_t move_down(int *matriz, int ancho, int alto) {
	cudaError_t cudaStatus;

	int *dev_m;	//Establecemos la matriz donde se van a recoger los resultados

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en setdevice");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_m, ancho*alto * sizeof(int)); //Reservamos memoria para la matriz resultado
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en Malloc");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_m, matriz, ancho*alto * sizeof(int), cudaMemcpyHostToDevice); //copiamos los datos iniciales
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Establecemos las dimensiones pertinentes
	dim3 dimgrid(alto, 1);
	dim3 dimblock(1, ancho, 1);

	mov_downK << < dimgrid, dimblock >> > (dev_m, ancho, alto);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error en synchronize mov_down\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en mov_down");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matriz, dev_m, ancho*alto * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en memcpy to host de mov_downK");
		goto Error;
	}

Error: //Liberamos la memoria de la variable en caso de error
	cudaFree(dev_m);

	return cudaStatus;
}

/*	add_left
*	Función del kernel para sumar hacia la izquierda todos los números que sean iguales.
*/
__device__ void add_left(int *matriz, int x, int y, int altura, int anchura)
{
	if (y != 0 && y < anchura)	//Los primeros hilos de la izquierda no deben realizar ninguna operacion pues serán modificados por los demas

	{
		if (matriz[x*anchura + y] != 0)	//Si es distinto de 0, gestiona su posible suma o desplazamiento
		{
			if (matriz[x*anchura + y] == matriz[x*anchura + (y - 1)]) //Si es igual a su vecino izquierdo, se procede a comprobar el numero de celdas con el mismo numero que hay en esa columna
			{
				int iguales = 0;
				iguales++;
				for (int i = 1; i <= y; i++)
				{
					if (matriz[x*anchura + y] == matriz[x*anchura + (y - i)])
					{
						iguales++;
					}
					else {
						break;
					}
				}
				if (iguales % 2 == 0)	//Si el numero es par, se suman, si no, ese numero será mezclado con otro y no estará disponible
				{
					matriz[x*anchura + (y - 1)] = matriz[x*anchura + (y - 1)] * 2;
					matriz[x*anchura + y] = 0;
				}
			}
			else if (matriz[x*anchura + (y - 1)] == 0)	 //Se comprueba que otros hilos hayan dejado 0 en sus operaciones para desplazarse
			{
				matriz[x*anchura + (y - 1)] = matriz[x*anchura + y];
				matriz[x*anchura + y] = 0;
			}
		}
	}
}

/*	stack_left
*	Función del kernel para desplazar todos los números hacia la izquierda.
*/
__device__ void stack_left(int *matriz, int anchura, int altura, int x, int y) {

	for (int i = anchura - 1; i > 0; i--)	 //realizaremos el desplazamiento celda a celda una altura-1 veces para gestionar la posibilidad del ultimo poniendose el primero de la lista
	{
		if ((y != 0) && (matriz[x*anchura + y] != 0) && matriz[x*anchura + (y - 1)] == 0)	//Si la celda pertenece a la primera fila, es 0 o su superior no es 0, no hace nada
		{
			matriz[x*anchura + (y - 1)] = matriz[x*anchura + y];	//Si lo es, desplazamos la celda
			matriz[x*anchura + y] = 0;
		}
		__syncthreads();	//utilizamos una sincronizacion para que estos pasos sean realizados a la vez por los hilos del bloque 
	}
}


/*	mov_leftK
*	Kernel que gestiona las operaciones para mover hacia la izquierda los numeros, sumandolos en el proceso
*/
__global__ void mov_leftK(int *matriz, int anchura, int altura) {
	int x = blockIdx.x;
	int y = threadIdx.y;

	stack_left(matriz, anchura, altura, x, y);		//Realizamos las llamadas de la siguiente manera para gestionar el movimiento:
	add_left(matriz, x, y, altura, anchura);		//2 2 0 4   -> 4 4 0 0
	__syncthreads();
	stack_left(matriz, anchura, altura, x, y);
}

/*	move_left
*	Metodo que gestiona la llamada al kerner mov_leftK
*/
cudaError_t move_left(int *matriz, int ancho, int alto) {
	cudaError_t cudaStatus;

	int *dev_m;		//Establecemos la matriz donde se van a recoger los resultados

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en setdevice");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_m, ancho*alto * sizeof(int));	//Reservamos memoria para la matriz resultado
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en Malloc");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_m, matriz, ancho*alto * sizeof(int), cudaMemcpyHostToDevice);	//copiamos los datos iniciales
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Establecemos las dimensiones pertinentes
	dim3 dimgrid(alto, 1);
	dim3 dimblock(1, ancho, 1);

	mov_leftK << < dimgrid, dimblock >> > (dev_m, ancho, alto);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error en synchronize mov_leftK\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en mov_leftK");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matriz, dev_m, ancho*alto * sizeof(int), cudaMemcpyDeviceToHost);	//recogemos el resultado en la variable del host
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en memcpy to host de mov_leftK");
		goto Error;
	}

Error:	//Liberamos la memoria de la variable en caso de error
	cudaFree(dev_m);

	return cudaStatus;
}

/*	add_right
*	Función del kernel para sumar hacia la izquierda todos los números que sean iguales.
*/
__device__ void add_right(int *matriz, int x, int y, int altura, int anchura)
{
	if (y != anchura - 1 && y < anchura)	//Los primeros hilos de la derecha no deben realizar ninguna operacion pues serán modificados por los demas
	{
		if (matriz[x*anchura + y] != 0)	//Si es distinto de 0, gestiona su posible suma o desplazamiento
		{
			if (matriz[x*anchura + y] == matriz[x*anchura + (y + 1)])//Si es igual a su superior, se procede a comprobar el numero de celdas con el mismo numero que hay en esa columna
			{
				int iguales = 0;
				iguales++;
				for (int i = 1; y + i < anchura; i++)
				{
					if (matriz[x*anchura + y] == matriz[x*anchura + (y + i)])
					{
						iguales++;
					}
					else {
						break;
					}
				}
				if (iguales % 2 == 0)		//Si el numero es par, se suman, si no, ese numero será mezclado con otro y no estará disponible
				{
					matriz[x*anchura + (y + 1)] = matriz[x*anchura + (y + 1)] * 2;
					matriz[x*anchura + y] = 0;
				}
			}
			else if (matriz[x*anchura + (y + 1)] == 0)	//Se comprueba que otros hilos hayan dejado 0 en sus operaciones para desplazarse
			{
				matriz[x*anchura + (y + 1)] = matriz[x*anchura + y];
				matriz[x*anchura + y] = 0;
			}
		}
	}
}

/*	stack_right
*	Función del kernel para desplazar todos los números hacia la derecha.
*/
__device__ void stack_right(int *matriz, int anchura, int altura, int x, int y) {

	for (int i = anchura - 1; i > 0; i--)	//realizaremos el desplazamiento celda a celda una altura-1 veces para gestionar la posibilidad del ultimo poniendose el primero de la lista
	{
		if ((y != anchura - 1) && (matriz[x*anchura + y] != 0) && matriz[x*anchura + (y + 1)] == 0)	//Si la celda pertenece a la primera fila, es 0 o su superior no es 0, no hace nada
		{
			matriz[x*anchura + (y + 1)] = matriz[x*anchura + y];	//Si lo es, desplazamos la celda
			matriz[x*anchura + y] = 0;
		}
		__syncthreads();	//utilizamos una sincronizacion para que estos pasos sean realizados a la vez por los hilos del bloque
	}
}

/*	mov_rightK
*	Kernel que gestiona las operaciones para mover hacia la derecha los numeros, sumandolos en el proceso
*/
__global__ void mov_rightK(int *matriz, int anchura, int altura) {
	int x = blockIdx.x;
	int y = threadIdx.y;

	stack_right(matriz, anchura, altura, x, y);		//Realizamos las llamadas de la siguiente manera para gestionar el movimiento:
	add_right(matriz, x, y, altura, anchura);		//2 2 0 4   -> 4 4 0 0
	__syncthreads();
	stack_right(matriz, anchura, altura, x, y);
}


/*	move_right
*	Metodo que gestiona la llamada al kerner mov_rightK
*/
cudaError_t move_right(int *matriz, int ancho, int alto) {	//Establecemos la matriz donde se van a recoger los resultados
	cudaError_t cudaStatus;

	int *dev_m;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en setdevice");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_m, ancho*alto * sizeof(int));	//Reservamos memoria para la matriz resultado
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en Malloc");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_m, matriz, ancho*alto * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Establecemos las dimensiones pertinentes
	dim3 dimgrid(alto, 1);
	dim3 dimblock(1, ancho, 1);

	mov_rightK << < dimgrid, dimblock >> > (dev_m, ancho, alto);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error en synchronize mov_right\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en mov_right");
		goto Error;
	}

	cudaStatus = cudaMemcpy(matriz, dev_m, ancho*alto * sizeof(int), cudaMemcpyDeviceToHost); //recogemos el resultado en la variable del host

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Error en memcpy to host de mov_rightK");
		goto Error;
	}

Error:	//Liberamos la memoria de la variable en caso de error
	cudaFree(dev_m);

	return cudaStatus;
}


int main()
{
	cudaError_t cudaStatus;
	srand(time(NULL));

	int ancho;
	int alto;
	int numSemillas = 0;
	int vidas = 5;
	char modo;
	char cargado;
	char ia;
	int *datos;
	int *matriz;
	int *especificaciones;

	especificaciones = MostrarEspecificaciones();

	printf("Desea activar la IA? (y/n): ");
	cin >> ia;

	printf("Desea comprobar si hay partidas guardadas?(y/n): ");
	cin >> cargado;
	if (cargado == 'y')
	{
		datos = cargar();

		vidas = datos[0];
		alto = datos[1];
		ancho = datos[2];

		int dificultad = datos[3];

		if (dificultad == 0)
		{
			modo = 'B';
			numSemillas = 15;
		}
		else
		{
			modo = 'A';
			numSemillas = 8;
		}

		matriz = (int*)malloc(ancho*alto * sizeof(int));

		for (int i = 0; i < alto*ancho; i++)
		{
			matriz[i] = datos[4 + i];
		}
	}
	else
	{
		printf("Indique el ancho de la matriz: ");
		cin >> ancho;
		printf("Indique el alto de la matriz: ");
		cin >> alto;

		
		if (ancho > especificaciones[0] || alto > especificaciones[0])
		{
			printf("La matriz seleccionada es demasiado grande para tu tarjeta grafica. Lo siento.");
			return 0;
		}
		

		printf("Indique la dificultad del juego (B->Bajo / A->Alto): ");
		cin >> modo;
		switch (modo)
		{
		case 'B':
			numSemillas = 15;
			break;
		case 'A':
			numSemillas = 8;
			break;
		default:
			break;
		}



		matriz = (int*)malloc(ancho*alto * sizeof(int));
		for (int i = 0; i < ancho*alto; i++) {
			matriz[i] = 0;
		}
	}

	if (ia == 'n')
	{
		while ((!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
		{
			system("CLS");

			if (!(!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
			{
				for (int i = 0; i < ancho*alto; i++) {
					matriz[i] = 0;
				}
				vidas--;
			}

			gestionSemillas(matriz, ancho, numSemillas, alto, modo);

			char movimiento = 'p';
			printf("Vidas restantes: %d\n", vidas);
			printf("Tablero:\n");
			showMatriz(matriz, ancho, alto);
			printf("Hacia donde quieres mover?(w/a/s/d) Para guardar teclee g: ");
			cin >> movimiento;
			switch (movimiento)
			{
			case 'w':
				cudaStatus = move_up(matriz, ancho, alto);
				break;
			case 'a':
				cudaStatus = move_left(matriz, ancho, alto);
				break;
			case 's':
				cudaStatus = move_down(matriz, ancho, alto);
				break;
			case 'd':
				cudaStatus = move_right(matriz, ancho, alto);
				break;
			case 'g':
				guardar(vidas, matriz, alto, ancho, modo);
				printf("Partida guardada, hasta pronto!");
				return 0;
			default:
				break;
			}

			if (!(!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
			{
				for (int i = 0; i < ancho*alto; i++) {
					matriz[i] = 0;
				}
				vidas--;
			}

		}

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
	else {

		while ((!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
		{
			if (!(!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
			{
				for (int i = 0; i < ancho*alto; i++) {
					matriz[i] = 0;
				}
				vidas--;
			}


			system("CLS");

			gestionSemillas(matriz, ancho, numSemillas, alto, modo);

			char movimiento = 'p';
			printf("Vidas restantes: %d\n", vidas);
			printf("Tablero:\n");
			showMatriz(matriz, ancho, alto);

			int r = rand() % 4;

			switch (r)
			{
			case 0:
				printf("Moviendo hacia arriba\n");
				cudaStatus = move_up(matriz, ancho, alto);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "move_up failed!");
					return 1;
				}
				break;
			case 1:
				printf("Moviendo hacia izquierda\n");
				cudaStatus = move_left(matriz, ancho, alto);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "move_left failed!");
					return 1;
				}
				break;
			case 2:
				printf("Moviendo hacia abajo\n");
				cudaStatus = move_down(matriz, ancho, alto);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "move_down failed!");
					return 1;
				}
				break;
			case 3:
				printf("Moviendo hacia derecha\n");
				cudaStatus = move_right(matriz, ancho, alto);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "move_right failed!");
					return 1;
				}
				break;
			default:
				break;
			}
			//Sleep(100);

			if (!(!checkFull(matriz, ancho*alto) || checkMove(matriz, ancho, alto)) && vidas > 0)
			{
				for (int i = 0; i < ancho*alto; i++) {
					matriz[i] = 0;
				}
				vidas--;
			}

		}



		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}


	return 0;
}

/*	showMatriz
*	Función que muestra la matriz por pantalla.
*/
void showMatriz(int *matriz, int anchura, int altura)
{
	for (int i = 0; i < altura; i++)
	{
		for (int j = 0; j < anchura; j++)
		{
			printf("%d\t", matriz[i*anchura + j]);
		}
		printf("\n");
	}
}

/*	generateSeeds
*	Función que genera una cantidad de semillas en la matriz, teniendo en cuenta sus dimensiones y en el modo de la dificultad que se encuentre. 
*	Si es nivel bajo “B” entonces se crearán 15 semillas con los valores 2, 4 y 8. Si es nivel alto “A” se crearán 8 semillas con valores 2, 4
*/
void generateSeeds(int *matriz, int ancho, int alto, int cantidad, char modo)
{
	int total = ancho * alto;
	int num;

	if (modo == 'B')
	{

		for (int i = 0; i < cantidad; i++)
		{
			int r = rand() % total;
			while (matriz[r] != 0) {
				r = rand() % total;
			}

			int opcion = rand() % 100;
			if (opcion <= 50) {
				matriz[r] = 2;
			}
			else if (opcion <= 80 && opcion > 50) {
				matriz[r] = 4;
			}
			else {
				matriz[r] = 8;
			}
		}
	}
	else if (modo == 'A')
	{
		for (int i = 0; i < cantidad; i++)
		{
			int r = rand() % total;
			while (matriz[r] != 0) {
				r = rand() % total;
			}

			int opcion = rand() % 100;
			if (opcion <= 60) {
				matriz[r] = 2;
			}
			else {
				matriz[r] = 4;
			}

		}
	}



}
/*	checkMove
*	Función que gestiona si se pueden realizar movimientos o no, esto servirá por si aunque la matriz este llena,
*	si se pueden realizar movimientos entonces no se acabe la partida. Para ello el método mirara en todas las
*	direcciones del eje cartesiano para ver si algún número es igual que el y se puede sumar o en cambio, si es un 0, desplazarse por la matriz.
*/
bool checkMove(int *matriz, int anchura, int altura)
{
	for (int i = 0; i < anchura*(altura - 1); i++)
	{
		if (matriz[i] == matriz[i + anchura] || matriz[i + anchura] == 0)
		{
			return true;
		}
	}

	for (int i = anchura; i < anchura*altura; i++)
	{
		if (matriz[i] == matriz[i - anchura] || matriz[i - anchura] == 0)
		{
			return true;
		}
	}

	for (int i = 0; i < altura; i++)
	{
		for (int j = 0; j < anchura - 1; j++)
		{
			if (matriz[i*anchura + i] == matriz[i*anchura + i + 1] || matriz[i*anchura + i + 1] == 0)
			{
				return true;
			}
		}
	}

	for (int i = 0; i < altura; i++)
	{
		for (int j = 1; j < anchura; j++)
		{
			if (matriz[i*anchura + i] == matriz[i*anchura + i - 1] || matriz[i*anchura + i - 1] == 0)
			{
				return true;
			}
		}
	}

	return false;

}
/*	checkFull
*	Función que gestiona si la matriz esta llena o no, es decir, si tiene algún 0 aún o no lo tiene. 
*/
int checkFull(int *matriz, int tamano)
{
	int flag = 1;
	for (int i = 0; i < tamano; i++)
	{
		if (matriz[i] == 0)
		{
			flag = 0;
		}
	}
	return flag;
}

/*	gestionSemillas
*	Función que gestiona cuántos huecos libres hay en la matriz mediante un contador, para llamar posteriormente a generateSeeds para crear las semillas necesarias, 
*	controlando en todo momento que el número de semillas a generar tengan hueco libre en la matriz.
*/
void gestionSemillas(int *matriz, int ancho, int numeroSemillas, int alto, char modo)
{
	if (!checkFull(matriz, ancho*alto))
	{
		int n = 0;
		for (int i = 0; i < ancho*alto; i++)
		{
			if (matriz[i] == 0)
				n++;
		}
		if (modo == 'B')
		{
			if (n < 15)
			{
				generateSeeds(matriz, ancho, alto, n, modo);
			}
			else {
				generateSeeds(matriz, ancho, alto, numeroSemillas, modo);
			}

		}
		else if (modo == 'A')
		{
			if (n < 8)
			{
				generateSeeds(matriz, ancho, alto, n, modo);
			}
			else {
				generateSeeds(matriz, ancho, alto, numeroSemillas, modo);
			}

		}

	}
}
/*	guardar
*	Función encargada de guardar la partida en un archivo externo (.txt) para preservar en el tiempo la partida por si se desea reanudarla más tarde desde en el punto que se quedo.
*/
void guardar(int vidas, int *matriz, int altura, int anchura, char dificultad) {

	ofstream archivo;
	int dif;

	archivo.open("2048_savedata.txt", ios::out); //Creamos o reemplazamos el archivo

	//Si no se puede guardar ERROR
	if (archivo.fail())
	{
		cout << "Error al guardar la partida.\n";
		exit(1);
	}

	if (dificultad == 'B')
	{
		dif = 0;
	}
	else
	{
		dif = 1;
	}

	archivo << vidas << endl; //Guardamos las vidas
	archivo << altura << endl; //Guardamos las altura
	archivo << anchura << endl; //Guardamos las anchura
	archivo << dif << endl; //Guardamos la dificultad

	//Guardamos la matriz
	for (int i = 0; i < (altura*anchura); i++)
	{
		archivo << matriz[i] << " ";
	}
	cout << "\nPartida guardada con exito." << endl;

	archivo.close(); //Cerramos el archivo
}
/*	cargar
*	Función que cargará la partida desde un archivo externo (.dat) en el vector de la matriz para proseguir jugando.
*/
int* cargar() {

	ifstream archivo;
	int i = 4, vidas, altura, anchura, dif;
	int *partida;

	archivo.open("2048_savedata.txt", ios::in); //Abrimos el archivo en modo lectura

	//Si no se puede cargar ERROR
	if (archivo.fail())
	{
		cout << "Error al abrir la partida guardada. El fichero no existe o está corrupto\n";
		exit(1);
	}

	archivo >> vidas;
	archivo >> altura;
	archivo >> anchura;
	archivo >> dif;

	partida = (int*)malloc(4 * sizeof(int) + altura * anchura * sizeof(int)); //Reservamos memoria para los datos de la partida

	partida[0] = vidas; //Guardamos vidas
	partida[1] = altura; //Guardamos altura
	partida[2] = anchura; //Guardamos anchura
	partida[3] = dif; //Guardamos la dificultad

	//Guardamos la matriz
	while (!archivo.eof()) { //Mientras no sea el final del archivo
		archivo >> partida[i];
		i++;
	}

	archivo.close(); //Cerramos el archivo

	return partida;
}

int* MostrarEspecificaciones()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int * especificacion;

	especificacion = (int*)malloc(2 * sizeof(int));
	for (int i = 0; i < 2; i++) {
		especificacion[i] = 0;
	}

	especificacion[0] = prop.maxThreadsPerBlock;
	especificacion[1] = *prop.maxGridSize;

	return especificacion;
}
