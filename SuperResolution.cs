using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.ColorSpaces;
using SixLabors.ImageSharp.ColorSpaces.Conversion;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;
using Size = SixLabors.ImageSharp.Size;

namespace ONNX.FastNeuralStyleTransfer;

public static class SuperResolution
{
    const int Width = 224;
    const int Height = 224;
    const int Channels = 1;

    public static Image<Rgb24> Process(Image<Rgb24> original)
    {
        var image = original.Clone(ctx =>
        {
            ctx.Resize(new ResizeOptions
            {
                Size = new Size(Width, Height),
                Mode = ResizeMode.Crop
            });
        });

        var input = new DenseTensor<float>(new[] {1, Channels, Height, Width});
        var converter = new ColorSpaceConverter();
        for (var y = 0; y < image.Height; y++)
        {
            var pixelSpan = image.DangerousGetPixelRowMemory(y);
            for (int x = 0; x < image.Width; x++)
            {
                var pixel = pixelSpan.Span[x];
                var rgb = new Rgb(pixel.R, pixel.G, pixel.B);
                var lab = converter.ToLms(rgb);
                input[0, 0, y, x] = lab.L;
            }
        }

        using var session = new InferenceSession(@"./model/super-resolution-10.onnx");
        
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results
            = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            });
        
        if (results.FirstOrDefault()?.Value is not Tensor<float> output)
            throw new ApplicationException("Unable to process image");

        var result = image.Clone(ctx =>
        {
            ctx.Resize(672, 672, new BicubicResampler());
        });
        
        for (var y = 0; y < result.Height; y++)
        {
            for (var x = 0; x < result.Width; x++)
            {
                var yCbCr = converter.ToYCbCr(result[x, y]);
                var pixel = new YCbCr(output[0, 0, y, x], yCbCr.Cb, yCbCr.Cr);
                result[x, y] = converter.ToRgb(pixel);
            }
        }

        return result;
    } 
}