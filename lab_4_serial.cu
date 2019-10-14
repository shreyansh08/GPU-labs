#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<time.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include <device_functions.h>

#define MASK_WIDTH 3    //Here MASK_WIDTH = MASK_HEIGHT = 2*N + 1 where N is half-width of the chosen square mask

unsigned char* readImg(char *filename,int *height_out, int *width_out, int *maxval_out)
{
    FILE *fptr;
    char buf[16];
    
    int i,j,k;
    int height,width,maxval;
    unsigned char *input_img;

    int index = 0;
    fptr =  fopen(filename,"rb");
    
    if(!fptr)
    {
        printf("Unable to open file '%s'\n",filename);
        return NULL;
    }
   
    if (!fgets(buf, sizeof(buf), fptr))
    {
        printf("Error reading format\n");
        return NULL;
    }

    int c = getc(fptr);
    while(c == '#'){
    while(getc(fptr)!='\n');
         c = getc(fptr);
    }
    ungetc(c, fptr);
	
    if(fscanf(fptr,"%d %d",&height,&width) !=2){
	printf("ERROR Reading Dimension\n");
	return NULL;
    }

    if(fscanf(fptr,"%d",&maxval)!=1){
	printf("ERROR Reading MAXDEPTH\n");
	return NULL;
    }
    
    while(fgetc(fptr) != '\n');
           
    printf("%d\t%d\t%d\n",height,width,maxval);
    
    int pix = width*height;

    input_img = (unsigned char*)(malloc((3*pix)*sizeof(unsigned char)));

    
    if (fread(input_img,sizeof(unsigned char),3*pix, fptr) != 3*pix)
    {
         printf("Error loading image '%s'\n", filename);
         return NULL;
    }

    *height_out = height;
    *width_out = width;
    *maxval_out = maxval;

    fclose(fptr);
    printf("Image read successfully\n");
    return input_img;
}

int writeImg(int width, int height, int maxval, unsigned char *output_img)
{
    FILE *fptr;
    fptr = fopen("output.ppm","wb");
    if(!fptr)
    {
        printf("Error opening file\n");
        return 0;
    }
    fprintf(fptr,"P6\n");

    fprintf(fptr, "%d %d\n",height,width);

    // rgb component depth
    fprintf(fptr, "%d\n",maxval);

    int pix = 3*width*height;
    // pixel data
    int j;
    if((j=fwrite(output_img,sizeof(unsigned char),pix, fptr))!=pix)
    {
	printf("ERROR WRITING %d\n",j);
    }
    fclose(fptr);
    return 1;
}

int main(int argc, char **argv)
{
    int height,width,maxval;
    unsigned char *input_img,*output_img;

    input_img = readImg(argv[1], &height, &width, &maxval);

    //printf("%d %d %d\n",height,width,maxval);
    
    int pix = width*height;

    output_img = (unsigned char*)(malloc(3*pix*sizeof(unsigned char)));

    int x,y,row,col,chan,i,j;
    unsigned char pixval;
    int freq[256];

    clock_t start_time = clock();
        
    for(chan=0;chan<3;chan++)
    {       
        for(row=0;row<height;row++)
        {
            for(col=0;col<width;col++)
            {   
                for(i=0;i<256;i++)freq[i]=0;

                for(x=row-MASK_WIDTH/2;x<=row+MASK_WIDTH/2;x++)
                {
                    for(y=col-MASK_WIDTH/2;y<=col+MASK_WIDTH/2;y++)
                    {
                        if((x >= 0) && (x < height) && (y>=0) && (y < width))
                        {
                            pixval = input_img[(x*width + y)*3 + chan];  
                        }
                        else
                        {
                            if(x<0 && y<0)
                            {
                                pixval = input_img[chan];
                            }
                            else if(x<0 && y<width)
                            {
                                pixval = input_img[3*y + chan];
                            }	
                            else if(x<0)
                            {
                                pixval = input_img[3*(width-1) + chan];
                            }
                            else if(x<height && y<0)
                            {
                                pixval = input_img[x*width*3 + chan];
                            }
                            else if(x<height && y>width)
                            {
                                pixval = input_img[(x*width +width-1)*3 + chan];
                            }
                            else if(x>height && y<0)
                            {
                                pixval = input_img[width*(height-1)*3 + chan];
                            }
                            else if(x>height && y<width)
                            {
                                pixval = input_img[(width*(height-1)+y)*3 + chan];
                            }
                            else
                            {
                                pixval = input_img[(width*(height-1) + (width-1))*3 + chan];
                            }
                        }
                        freq[pixval]++;
                    }
                }

                j=0;
                for(i=0;i<256;i++)
                {
                    j=j+freq[i];
                    if(j>((MASK_WIDTH*MASK_WIDTH)/2))break;    
                }
    
                output_img[(row*width + col)*3 + chan] = i;
            }
        }
    }

    clock_t end_time = clock();

    float serial_time = ((double)(end_time-start_time)*1000)/CLOCKS_PER_SEC;


    i = writeImg(width,height,maxval,output_img);
    if(i==0)
    return 1;
    
    printf("%f\n",serial_time);
    
}