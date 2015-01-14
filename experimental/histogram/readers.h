/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cstdio>
#include <fstream>

//---------------------------------------------------------------------
// Targa .tga image file parsing
//---------------------------------------------------------------------

/**
 * TGA image header info
 */
struct TgaHeader
{
    char idlength;
    char colormaptype;
    char datatypecode;
    short colormaporigin;
    short colormaplength;
    char colormapdepth;
    short x_origin;
    short y_origin;
    short width;
    short height;
    char bitsperpixel;
    char imagedescriptor;

    void Parse (FILE *fptr)
    {
        idlength = fgetc(fptr);
        colormaptype = fgetc(fptr);
        datatypecode = fgetc(fptr);
        fread(&colormaporigin, 2, 1, fptr);
        fread(&colormaplength, 2, 1, fptr);
        colormapdepth = fgetc(fptr);
        fread(&x_origin, 2, 1, fptr);
        fread(&y_origin, 2, 1, fptr);
        fread(&width, 2, 1, fptr);
        fread(&height, 2, 1, fptr);
        bitsperpixel = fgetc(fptr);
        imagedescriptor = fgetc(fptr);
    }

    void Display (FILE *fptr)
    {
        fprintf(fptr, "ID length:         %d\n", idlength);
        fprintf(fptr, "Color map type:    %d\n", colormaptype);
        fprintf(fptr, "Image type:        %d\n", datatypecode);
        fprintf(fptr, "Color map offset: %d\n", colormaporigin);
        fprintf(fptr, "Color map length: %d\n", colormaplength);
        fprintf(fptr, "Color map depth:  %d\n", colormapdepth);
        fprintf(fptr, "X origin:          %d\n", x_origin);
        fprintf(fptr, "Y origin:          %d\n", y_origin);
        fprintf(fptr, "Width:             %d\n", width);
        fprintf(fptr, "Height:            %d\n", height);
        fprintf(fptr, "Bits per pixel:    %d\n", bitsperpixel);
        fprintf(fptr, "Descriptor:        %d\n", imagedescriptor);
    }
};


/**
 * Decode image byte data into pixel
 */
void ParseTgaPixel(uchar4 &pixel, unsigned char *tga_pixel, int bytes)
{
    if (bytes == 4)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = tga_pixel[3];
    }
    else if (bytes == 3)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = 0;
    }
    else if (bytes == 2)
    {
        pixel.x = (tga_pixel[1] & 0x7c) << 1;
        pixel.y = ((tga_pixel[1] & 0x03) << 6) | ((tga_pixel[0] & 0xe0) >> 2);
        pixel.z = (tga_pixel[0] & 0x1f) << 3;
        pixel.w = (tga_pixel[1] & 0x80);
    }
}


/**
 * Reads a .tga image file
 */
void ReadTga(uchar4* &pixels, int &width, int &height, const char *filename)
{
    // Open the file
    FILE *fptr;
    if ((fptr = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "File open failed\n");
        exit(-1);
    }

    // Parse header
    TgaHeader header;
    header.Parse(fptr);
//    header.Display(stdout);
    width = header.width;
    height = header.height;

    // Verify compatibility
    if (header.datatypecode != 2 && header.datatypecode != 10)
    {
        fprintf(stderr, "Can only handle image type 2 and 10\n");
        exit(-1);
    }
    if (header.bitsperpixel != 16 && header.bitsperpixel != 24 && header.bitsperpixel != 32)
    {
        fprintf(stderr, "Can only handle pixel depths of 16, 24, and 32\n");
        exit(-1);
    }
    if (header.colormaptype != 0 && header.colormaptype != 1)
    {
        fprintf(stderr, "Can only handle color map types of 0 and 1\n");
        exit(-1);
    }

    // Skip unnecessary header info
    int skip_bytes = header.idlength + (header.colormaptype * header.colormaplength);
    fseek(fptr, skip_bytes, SEEK_CUR);

    // Read the image
    int             pixel_bytes     = header.bitsperpixel / 8;

    // Allocate and initialize pixel data
    if (pixels == NULL)
    {
        if ((pixels = (uchar4*) malloc(width * height * sizeof(uchar4))) == NULL)
        {
            fprintf(stderr, "malloc of image failed\n");
            exit(-1);
        }
    }
    memset(pixels, 0, header.width * header.height * sizeof(uchar4));

    // Parse pixels
    unsigned char   tga_pixel[5];
    int             current_pixel = 0;
    while (current_pixel < header.width * header.height)
    {
        if (header.datatypecode == 2)
        {
            // Uncompressed
            if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d  (uncompressed)\n", current_pixel);
                exit(-1);
            }
            ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
            current_pixel++;
        }
        else if (header.datatypecode == 10)
        {
            // Compressed
            if (fread(tga_pixel, 1, pixel_bytes + 1, fptr) != pixel_bytes + 1)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d (compressed)\n", current_pixel);
                exit(-1);
            }
            int run_length = tga_pixel[0] & 0x7f;
            ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
            current_pixel++;

            if (tga_pixel[0] & 0x80)
            {
                // RLE chunk
                for (int i = 0; i < run_length; i++)
                {
                    ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
                    current_pixel++;
                }
            }
            else
            {
                // Normal chunk
                for (int i = 0; i < run_length; i++)
                {
                    if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
                    {
                        fprintf(stderr, "Unexpected end of file at pixel %d (normal)\n", current_pixel);
                        exit(-1);
                    }
                    ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
                    current_pixel++;
                }
            }
        }
    }

    // Close file
    fclose(fptr);
}


//---------------------------------------------------------------------
// Binary .bin image file parsing
//---------------------------------------------------------------------

/**
 * Reads a .bin image file comprised of a straightforward, uncompressed serialization of float4 pixel data
 */
void ReadBin(float4* &pixels, int &width, int &height, const char *filename)
{
    // Open file
    FILE *fptr;
    if ((fptr = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "File open failed\n");
        exit(-1);
    }

    // Get file size
    fseek(fptr, 0, SEEK_END);
    int file_size = ftell(fptr);
    fseek(fptr, 0, SEEK_SET);

    if ((width == -1) || (height == -1))
    {
        // Set dims to 1 x file_size
        height = 1;
        width = file_size;
    }

    // Allocate float4
    if ((pixels = (float4*) malloc(width * height * sizeof(float4))) == NULL)
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }

    if (fread(pixels, 1, file_size, fptr) != file_size)
    {
        fprintf(stderr, "Unexpected end of file\n");
        exit(-1);
    }

    fclose(fptr);
}
