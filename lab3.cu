#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>

__global__ void rgbToGreyKernel(int height,int width ,unsigned char *input_img, unsigned char *output_img)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(row<height && col<width)
    {
        int idx = row*width + col;
        float red = (float)input_img[3*idx];
        float green = (float)input_img[3*idx+1];
        float blue = (float)input_img[3*idx+2];
        
        output_img[idx] = 0.21*red + 0.71*green + 0.07*blue;
    }

}


int main(int argc, char **argv)
{
    FILE *fptr;
    char buf[16];
    
    int i,j,k;
    
    int width, height, maxval;
    unsigned char *input_img,*output_img;
    unsigned char *dev_input_img,*dev_output_img;

    int index = 0;
    fptr =  fopen(argv[1],"rb");
    
    if(!fptr)
    {
        printf("Unable to open file '%s'\n",argv[1]);        
    }
   
    if (!fgets(buf, sizeof(buf), fptr))
    {
        printf("Error reading format\n");
        return 1;
    }

    //printf("Hi\n");    

    int c = getc(fptr);
    while(c == '#')
    {
        while(getc(fptr)!='\n');
        c = getc(fptr);
    }
    ungetc(c, fptr);
	
    if(fscanf(fptr,"%d %d",&height,&width) !=2)
    {
        printf("ERROR Reading Dimension\n");
        return 1;
    }

    if(fscanf(fptr,"%d",&maxval)!=1)
    {
        printf("ERROR Reading MAXDEPTH\n");
    	return 1;
    }
    
    while(fgetc(fptr) != '\n');
           
    printf("%d\t%d\t%d\n",height,width,maxval);
    
    int pix = width*height; //number of pixels

    input_img = (unsigned char*)(malloc((3*pix)*sizeof(unsigned char)));

    
    if (fread(input_img,sizeof(unsigned char),3*pix, fptr) != 3*pix)
    {
         printf("Error loading image '%s'\n", argv[1]);
         return 1;
    }

    cudaMalloc((void**)&dev_input_img,3*pix*sizeof(unsigned char));
    cudaMalloc((void**)&dev_output_img,pix*sizeof(unsigned char));
    
    if(input_img == NULL)
    {
        printf("ERROR\n");
    }

    cudaMemcpy(dev_input_img,input_img,3*pix*sizeof(unsigned char),cudaMemcpyHostToDevice);

    int T;

    for(T=1;T<128;T=T*2)
    {        
        dim3 DimGrid((width-1)/T + 1,(height-1)/T + 1,1);
        dim3 DimBlock(T,T,1);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);	
        
        cudaEventRecord(start);
        rgbToGreyKernel<<<DimGrid,DimBlock>>>(height,width,dev_input_img,dev_output_img);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
     
        float parallel_time = 0;
        cudaEventElapsedTime(&parallel_time, start, stop);
     
        output_img = (unsigned char*)(malloc(pix*sizeof(unsigned char)));

        clock_t start_time = clock();
        for(i=0;i<pix;i++)
        {
        output_img[i] = 0.21*input_img[3*i] + 0.71*input_img[3*i+1] + 0.07*input_img[3*i+2];
        }
        clock_t end_time = clock();

        float serial_time = ((double)(end_time-start_time)*1000)/CLOCKS_PER_SEC;

        printf("Threads per block(T*T) = %d*%d, Serial Time = %f, Parallel Time = %f, Speedup = %f\n",T,T,serial_time,parallel_time,serial_time/parallel_time);
    }

    cudaMemcpy(output_img,dev_output_img,pix*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    fclose(fptr);

    fptr = fopen("output.pgm","wb");
    if(!fptr)
    {
        printf("Error opening file\n");
         return 1;
    }
    fprintf(fptr,"P5\n");

    fprintf(fptr, "%d %d\n",height,width);

    // rgb component depth
    fprintf(fptr, "%d\n",maxval);

    // pixel data
    if((j=fwrite(output_img,sizeof(unsigned char),pix, fptr))!=pix)
    {
	    printf("ERROR WRITING %d\n",j);
    }
    fclose(fptr);

    cudaFree(dev_input_img);
    cudaFree(dev_output_img);
    free(input_img);
    free(output_img);
    //printf("%d %d %d %d\n",input_img[0],input_img[1],input_img[2],output_img[0]);
    //for(i=0;i<height;i++)printf("%d ",output_img[i]);
}