# Sobre formatos de imagen

## RAW (.raw)

-   Este fichero contiene los datos únicamente de píxeles en bruto, en el orden exacto en el que salen del dispositivo. No contiene metadatos (width, height, bit_depth...) sonre la imagen.

### Cámaras profesionales/industriales

-   Las cámaras industriales (Basler, FLIR, Allied Vision...) suelen generar: `image.raw` `image.ini` (o `.cfg`, `.txt`) donde se especifican los metadatos de la imagen cruda:

```ini
width=4096
height=2168
bit_depth=12
bayer_pattern=RGGB
endianess=little
```

### Sensores/sistemas embebidos

-   El fichero `.raw` suele venir sin metadatos. Las dimensiones se conocen porque típicamente son constantes en la aplicación
-   Las dimensiones de la imagen siempre suelen ser: `4096x2168 12-bit`

### Archivos RAW fotográficos - Formatos Propietarios

-   Llevan incluidos los metadatos en el propio fichero
-   Utilizar `rawpy` para su lectura
-   Ejemplos de formatos propietarios:
    -   .DNG Apple ProRAW Image
    -   .ARI ARRIRAW Image
    -   .CR2 Canon Raw 2 Image File
    -   .CR3 Canon Raw 3 Image File
    -   .CRW Canon Raw CIFF Image File
    -   .CS1 CaptureShop 1-shot Raw Image
    -   .BAY Casio RAW Image
    -   .DNG Digital Negative Image
    -   .EIP Enhanced Image Package File
    -   .ERF Epson RAW File
    -   .CXI FMAT RAW Image
    -   .RAF Fuji RAW Image File
    -   .GPR GoPro RAW Image
    -   .3FR Hasselblad 3F RAW Image
    -   .FFF Hasselblad RAW Image
    -   .KC2 Kodak DCS200 Camera Raw Image
    -   .DCR Kodak Digital Camera RAW Image File
    -   .K25 Kodak K25 Image
    -   .KDC Kodak Photo-Enhancer File
    -   .MOS Leaf Camera RAW File
    -   .RWL Leica RAW Image
    -   .MFW Mamiya Camera Raw File
    -   .MEF Mamiya RAW Image
    -   .MDC Minolta Camera Raw Image
    -   .MRW Minolta Raw Image File
    -   .NKSC Nikon Capture NX-D Sidecar File
    -   .NEF Nikon Electronic Format RAW Image
    -   .NRW Nikon Raw Image
    -   .ORF Olympus RAW File
    -   .RW2 Panasonic RAW Image
    -   .PEF Pentax Electronic File
    -   .IIQ Phase One RAW Image
    -   .RAW Raw Image Data File
    -   .RWZ Rawzor Compressed Image
    -   .J6I Ricoh Camera Image File
    -   .SRW Samsung RAW Image
    -   .X3F SIGMA X3F Camera RAW File
    -   .ARW Sony Alpha Raw Digital Camera Image
    -   .SR2 Sony RAW Image
    -   .SRF Sony RAW Image

### Datos científicos: microscopía, satélites...

-   Muy habitual que el fichero `.raw` venga acompañado de un fichero de metadatos `.meta`, `.hdr`, `.txt`, `.json`, `.xml`... donde se especifican las características de la imagen y demás detalles
