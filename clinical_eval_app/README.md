# Clinical Eval App

Aplicacion web estatica para validacion clinica de justificaciones de IA (Match / No Match), sin backend.

## Objetivo

Permitir que evaluadores medicos revisen casos en local, registren veredictos y exporten resultados en CSV para analisis posterior.

## Estructura esperada

- index.html
- style.css
- app.js
- data/cases.json
- data/images/

Tambien puedes trabajar con datasets alternativos (por ejemplo exportados por escenario) cargandolos manualmente desde el Gestor de Data.

## Esquema de entrada (cases.json)

Cada caso debe seguir este formato:

```json
[
  {
    "id_imagen": "polyp_042",
    "image_path": "data/images/polyp_042_bbox.jpg",
    "ground_truth_class": "AD",
    "id_modelo": "qwen3.5-9b",
    "clinical_justification": "La lesion presenta un patron vascular irregular..."
  }
]
```

Notas:

- id_modelo no se muestra en la UI (doble ciego), pero se usa en el CSV exportado.
- image_path debe ser una ruta valida desde index.html (normalmente relativa).

## Flujo principal

1. La app intenta cargar data/cases.json.
2. Muestra un caso por pantalla con imagen, diagnostico real y justificacion.
3. El evaluador vota Match o No Match y puede anadir comentario opcional.
4. Se puede navegar con flechas y abrir galeria para saltar a cualquier caso.
5. El progreso se guarda en localStorage.
6. El CSV se puede exportar durante la evaluacion o al finalizar.

## Pantalla inicial y carga local

Si no hay dataset activo, la app muestra la pantalla inicial de carga.

- Desde ahi se importa cases.json manualmente.
- Al importar, la app reemplaza el dataset en memoria/cache y reinicia el progreso para evitar mezclar sesiones.
- En esa pantalla no se muestra accion de borrado porque aun no hay un dataset activo.

## Gestor de Data

El Gestor de Data (boton "Gestor de Data" en la barra inferior) permite:

- Importar un nuevo cases.json.
- Borrar estado local desde cero.

Importante: "Borrar estado local" elimina cache de casos y progreso guardado localmente.

## Galeria interactiva

La galeria incluye filtros interactivos por estado:

- Match, No Match y Pendiente: filtro exclusivo (solo uno activo a la vez).
- Con comentario: filtro adicional combinable con cualquiera de los anteriores.

Las etiquetas de filtro tienen tamano homogeneo para mejorar lectura y consistencia visual.

## Manejo de data faltante o rota

Si la app detecta que la data actual fue borrada o que faltan imagenes referenciadas:

- limpia estado local,
- vuelve a la pantalla inicial de carga,
- muestra un mensaje de error explicando el problema.

Esto evita quedarse en un estado intermedio sin informacion visible.

## Persistencia local

Se guardan en localStorage:

- resultados acumulados,
- indice del caso actual,
- cache del dataset cargado.

Si se cierra y reabre la pagina, la evaluacion se reanuda cuando existe data valida en cache.

## Exportacion CSV

Cabecera exacta:

```text
id_imagen,tipo_polipo,id_modelo,veredicto,comentario_medico
```

Nombre de descarga:

- evaluacion_clinica_resultados.csv

## Uso rapido

1. Coloca imagenes en data/images/.
2. Genera o edita data/cases.json con el esquema correcto.
3. Abre index.html en el navegador.
4. Evalua casos y exporta CSV cuando quieras.