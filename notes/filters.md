## Conversión de formato y espacios de color

## Bayer -> RGB ✅

Implementación del demosaicing considerando distintos tamaños de patrón Bayer.
Resultado en imagen RGB real.

El resultado es una imagen donde cada canal es tipo real con rango 0-1

> [!NOTE] Todas las operaciones de transformación deben asumir el formato de imagen en ese rango y tipo. Puede ser en color de tres canales o en escala de grises con un solo canal

## Real -> RGB de 8 bits

Conversión desde real.

[0, 1] -> A formato RGB con 1 byte por canal.

## RGB -> HSV

## HSV -> RGB

## Color -> Gris

## Gris -> Color

## Transformadores

## Normalización min-max

## Normalización min-max con percentiles

Cálculo de min/max usando percentiles (p.ej. 1 y 99) para robustez frente a outliers.

## Ajuste gamma

```math
s = c * r^{gamma}
```

## Ajuste por función sigmoide ✅

## Rotaciones

## Flipping

Horizontal y/o vertical.

## Ecualización del histograma

## CLAHE (Contrast Limited Adaptive Histogram Equalization)

## Filtro gaussiano

Suavizado mediante convolución con un núcleo gaussiano con parámetro sigma.

## Filtro de mediana

## Filtro de paso alto / Realce de bordes (Unsharp Masking)

```math
I_{afilada} = I_{orignal} + peso * (I_{original} - I_{suavizada})
```
