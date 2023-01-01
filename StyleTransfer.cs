using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ONNX.FastNeuralStyleTransfer;

public static class StyleTransfer
{
    const int Width = 224;
    const int Height = 224;
    const int Channels = 3;

    public static Image<Rgb24> Process(Image<Rgb24> original, string model)
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
        for (var y = 0; y < image.Height; y++)
        {
            var pixelSpan = image.DangerousGetPixelRowMemory(y);
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = pixelSpan.Span[x];
                input[0, 0, y, x] = pixel.R;
                input[0, 1, y, x] = pixel.G;
                input[0, 2, y, x] = pixel.B;

            }
        }

        using var session = new InferenceSession($@"./model/{model}");
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results
            = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input1", input)
            });

        if (results.FirstOrDefault()?.Value is not Tensor<float> output)
            throw new ApplicationException("Unable to process image");

        var result = new Image<Rgb24>(Width, Height);
        for (var y = 0; y < Height; y++)
        {
            for (var x = 0; x < Width; x++)
            {
                result[x, y] = new Rgb24(
                    FloatPixelValueToByte(output[0, 0, y, x]),
                    FloatPixelValueToByte(output[0, 1, y, x]),
                    FloatPixelValueToByte(output[0, 2, y, x])
                );
            }
        }

        return result;
    }

    static byte FloatPixelValueToByte(float pixelValue) =>
        (byte) Math.Clamp(pixelValue, 0, 255);
}