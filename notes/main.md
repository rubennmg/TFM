# 18/11/2025

## Notes

-   [Basic Image Processing](https://www.cs.rice.edu/~vo9/recognition/notebooks/image_processing_lab.html)
-   PIL vs Rawpy

## TODOs

-   [x] Mejoras de escritura de código
-   [x] Utilizar método to `np_int8_array`
-   [x] Optimizar `image_loader`
    -   Nueva estructura para cargar diferentes formatos de imagen. Integración pendiente
-   [ ] `clamp_()` para disminuir cantidad de memoria reservada y copias de memoria
-   [ ] Optimizar tratamiento del tensor
-   [ ] Sesión de debug
-   [ ] Mirar HSV

# 19/11/2025

## Notes

## TODOs

-   [x] Carga de imágenes ARW
-   [x] Carga de imágenes DNG
-   [x] Carga de imágenes JPG
-   [x] Integración del nuevo `image_loader`

# 22/11/2025

## Notes

## TODOs

-   [x] Integración completa en `controller`
    -   [x] Revisión y debug
    -   [x] Mejora de tratamiento del tensor
    -   [x] Mejora en operaciones
        -   [x] `show_tensor` (`image_viewer`)
-   [x] Image model tiene un array de numpy int8 que se genera una vez se carga la imagen para visualización
-   [x] Reestructurar `utils`
-   [x] Simplificar reset de zoom + controles de transformadores
-   [x] Analizar actualización de histograma mediante `torch_bins`
-   [ ] Revisar signal rate en `right_panel`
-   [ ] Dilema de si actualizar o no el tensor original al debayerizar, esto genera un problema al aplicar mejora de contraste y desbayerizar
-   [ ] Mostrar nombre de la imagen
-   [ ] Mostrar RGB, RAW, quizá `shape`...
-   [ ] Desacoplamiento e implementación de nuevos filtros
-   [ ] Tests
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

# 23/11/2025

## Notes

## TODOs

-   [x] Revisar signal rate en `right_panel`
-   [x] Dilema de si actualizar o no el tensor original al debayerizar, esto genera un problema al aplicar mejora de contraste y desbayerizar
-   [x] Mostrar nombre de la imagen
-   [x] Mostrar RGB, RAW, quizá `shape`...
-   [x] Implementación de debayers
-   [x] Actualizar image info de forma dinámica
-   [ ] Tests de debayers
-   [ ] Desacoplamiento e implementación de nuevos filtros
-   [ ] Tests
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

# 24/11/2025

## Notes

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [ ] Desacoplamiento e implementación de nuevos filtros
    -   [x] Clase abstracta general `Image_Operation`
    -   [x] Implementación de Debayer extendiendo clase abstracta
    -   [x] Implementación de Mejora de Contraste con Sigmoide extendiendo clase abstracta
    -   [ ] Estructurar para el resto de filtros/transformadores
    -   [ ] Implementar siguiendo clase abstracta
-   [ ] Debuggear lo del reset de image tras aplicar operaciones mediante controles. Creo que me voy a tener que cargar el procesado "en tiempo real"
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

# 25/11/2025

## Notes

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [x] Desacoplamiento e implementación de nuevos filtros
    -   [x] Clase abstracta general `Image_Operation`
    -   [x] Implementación de Debayer extendiendo clase abstracta
    -   [x] Implementación de Mejora de Contraste con Sigmoide extendiendo clase abstracta
    -   [x] Estructurar para el resto de filtros/transformadores
    -   [x] Implementar siguiendo clase abstracta
-   [x] Debuggear lo del reset de image tras aplicar operaciones mediante controles. Creo que me voy a tener que cargar el procesado "en tiempo real"
-   [x] Creación de registro de operaciones para futuro benchmark
-   [ ] Implementación de operaciones CORE + GUI
-   [ ] Estructura de tests
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

## 26/11/2025

## Notes

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [x] Loaders - ABC
-   [ ] Implementación de operaciones CORE + GUI
-   [ ] Estructura de tests
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

## 28/11/2025

## Notes

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [x] Loaders - ABC
-   [ ] Implementación de operaciones CORE + GUI
    -   [x] Rotate
    -   [x] Flip
    -   [ ] ...
-   [ ] Estructura de tests
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

## 29/11/2025

## Notes

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [ ] Implementación de operaciones CORE + GUI
    -   [x] Rotate
    -   [x] Flip
    -   [x] Gaussian
    -   [x] MinMax
    -   [x] MinMax Percentile
    -   [x] ColorToGray
    -   [x] GrayToColor
    -   [x] RGBtoHSV
    -   [x] HSVtoRGB
    -   [ ] RealToRGB8
    -   [ ] RGB8ToReal
    -   [ ] Gamma
    -   [ ] CLAHE
    -   [ ] Histogram Equalization
    -   [ ] Unsharp Masking
-   [x] Collapsible sections en panel de operaciones
-   [ ] Estructura de tests
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen

## 30/11/2025

## Notes

-   Filtro de mediana satura mucho la memoria de la GPU -> revisar
-   Gestión de espacios de color en la aplicación de filtros
    -   Filtros sobre `tensor` o sobre `original_tensor`

## TODOs

-   [ ] Tests de debayers <-> standby de momento
-   [ ] Implementación de operaciones CORE + GUI
    -   [x] Rotate
    -   [x] Flip
    -   [x] Gaussian
    -   [x] MinMax
    -   [x] MinMax Percentile
    -   [x] ColorToGray
    -   [x] GrayToColor
    -   [x] RGBtoHSV
    -   [x] HSVtoRGB
    -   [x] RealToRGB8
    -   [x] RGB8ToReal
    -   [x] Gamma
    -   [ ] CLAHE
    -   [ ] Histogram Equalization
    -   [ ] Unsharp Masking
-   [x] Simplificar código de `right_panel`
-   [x] Reset de controles
-   [ ] Estructura de tests
-   [ ] Tests
-   [ ] Preparar diagrama de Arquitectura COMPLETO:
    -   `Image` -> `image_loader` -> `image_operation` -> `Controller` -> `GUI`
    -   IMAGE LOADERS - IMAGE FORMATS:
        -   `raw_loader`
        -   `rawpy_loader` - `rawpy` :
            -   `arw_loader`
            -   `dng_loader`
        -   `jpg_loader` - `PIL`
    -   GUI:
        -   `main_window`
            -   `left_panel`
                -   `rgb_histogram`
                -   `load_control`
                -   `reset_control`
            -   `image_viewer`
                -   `image_canvas` (scrollable)
                -   `image_info`
            -   `right_panel`
                -   `[operation_control]`
                -   `debayer_control`
-   [ ] Modelo para perfil de configuración - json. Interesante usar `pydantic`
-   [ ] Bug: ventana resize tras cargar imagen
