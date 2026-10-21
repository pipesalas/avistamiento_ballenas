# Producción con pocas visitas

## Arquitectura recomendada

La versión actual usa Vue/Vite (`frontend/`) y FastAPI (`api/`). Firebase
Hosting puede servir el frontend compilado y las fotografías desde su CDN;
Cloud Run ejecuta la API Python. Firebase Hosting por sí solo no ejecuta
`api/main.py`. Firebase App Hosting es otro producto y no hace falta para
este frontend estático.

```text
avistamientos.vuelvealoceano.cl
└── Firebase Hosting
    ├── /assets/*           → frontend/dist/assets
    ├── /media/*            → fotografías y miniaturas publicadas como estáticos
    ├── /api/**             → rewrite hacia Cloud Run
    └── otras rutas        → index.html (Vue Router)
```

Esta integración requiere un proyecto con facturación vinculada (Blaze).
El costo depende de las cuotas disponibles, región, transferencia, imágenes
de contenedor, builds y uso de la API; no existe un precio fijo por 1.000 visitas.
Las cuotas gratuitas pueden cubrir el cómputo de un proyecto pequeño, pero no
garantizan una factura total de cero.

La integración Hosting/Cloud Run evita añadir un balanceador dedicado para el
dominio personalizado. Hay que elegir una región de Cloud Run admitida por
los rewrites de Hosting. WordPress puede seguir en `vuelvealoceano.cl`.
Usar `/avistamiento_ballenas/` dentro de ese WordPress requiere configurar un
proxy en el hosting actual o delante de él y adaptar las rutas base de la app.

La configuración incluida usa `us-central1` tanto en los scripts de GCP como
en el rewrite de Firebase. `pinTag` vincula cada publicación de Hosting con
una revisión concreta de la API.

## Primera publicación: paso a paso

### 1. Guardar la versión en Git

Los cambios del editor/sandbox son archivos de trabajo hasta que se hace un
commit. Revisar los archivos modificados y nuevos, agregarlos al staging y
crear un commit en la rama de trabajo (`new-arch` en esta sesión). Después,
subir esa rama a `origin`. Un push guarda la versión en GitHub; **no ejecuta
automáticamente el despliegue**.

Si el sandbox es una copia aislada del repositorio local, el commit debe llegar
a GitHub o exportarse antes de descartar el sandbox. En el equipo local se
puede traer esa rama con `git fetch origin` y, si ya se está en `new-arch` con
el árbol de trabajo limpio, `git pull --ff-only origin new-arch`. Si la interfaz
del sandbox ofrece aplicar/exportar cambios, también se puede usar para llevar
los archivos al repositorio local y hacer allí el commit.

Los archivos `frontend/dist/`, las miniaturas, `node_modules/` y los entornos
Python se regeneran; el código, datos fuente, lockfile y configuración de
despliegue sí forman parte de la versión.

### 2. Crear el proyecto con la cuenta personal

1. Entrar en <https://console.firebase.google.com/> con la cuenta de Google
   que será propietaria del proyecto.
2. Crear un proyecto, por ejemplo **Avistamientos Vuelve al Océano**. Analytics
   puede quedar desactivado para este despliegue.
3. Vincular una cuenta de facturación y activar el plan **Blaze**.
4. Copiar el **Project ID** desde Configuración del proyecto → General.
   Se usa el identificador, no el nombre visible ni el número del proyecto.
5. En Firebase → Hosting, completar el inicio de Hosting si la consola lo pide.

No hace falta descargar claves de cuentas de servicio. La autenticación del
desarrollador se realiza en el navegador; las cuentas de ejecución/build se
crean dentro del proyecto.

### 3. Autenticarse desde el equipo donde se publicará

Instalar [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), Node.js 22
y [uv](https://docs.astral.sh/uv/getting-started/installation/). Desde la raíz
del repositorio:

```bash
gcloud auth login
npx --yes firebase-tools@15.34.0 login
export PROJECT_ID="ID_REAL_DEL_PROYECTO"
```

Elegir la misma cuenta personal en ambos inicios de sesión. Los comandos usan
`--project` explícitamente para no depender del proyecto predeterminado del equipo.
El usuario que ejecuta la preparación debe poder habilitar servicios, crear
cuentas y asignar permisos; la cuenta propietaria del proyecto nuevo puede hacerlo.

### 4. Preparar GCP y desplegar

Una vez por proyecto:

```bash
bash scripts/setup_gcp.sh "$PROJECT_ID"
```

Este script habilita las APIs, crea el repositorio Docker y dos cuentas de
servicio. La API no recibe roles de acceso a datos de GCP: lee archivos de su
imagen. La cuenta de build recibe permisos para leer el código subido a Storage,
publicar imágenes y escribir logs.

Para publicar:

```bash
bash scripts/deploy.sh "$PROJECT_ID"
```

El despliegue construye la imagen en Cloud Build, publica Cloud Run con 1 CPU,
1 GiB, concurrencia 8, mínimo 0 y máximo de servicio 2, y finalmente publica
Hosting. El predeploy de Firebase instala las dependencias del frontend desde
`package-lock.json`, compila Vue, genera miniaturas y copia los medios. Cloud
Build genera la imagen Linux AMD64, también si el equipo del desarrollador es
un Mac ARM. No hace falta Docker local para este flujo.

La imagen fija las dependencias Python mediante `requirements-api.lock`; el
archivo `requirements-api.txt` conserva los requisitos de origen. Al actualizar
dependencias hay que regenerar el lockfile, revisar los cambios y verificar la API.

### 5. Verificar y conectar el subdominio

Antes de conectar el dominio, abrir:

- `https://PROJECT_ID.web.app/`: mapa.
- `https://PROJECT_ID.web.app/api/meta`: respuesta JSON de Cloud Run.
- `https://PROJECT_ID.web.app/datos`: acceso directo a una ruta Vue.
- Una fecha con fotos: galería, visor ampliado y descarga CSV.

Luego, Firebase → Hosting → Agregar dominio personalizado:
**`avistamientos.vuelvealoceano.cl`**. Agregar en el administrador DNS los
registros exactos que entregue Firebase y esperar la verificación y el
certificado HTTPS. Solo se configura ese subdominio; el dominio raíz sigue
apuntando a WordPress.

### 6. Actualizaciones posteriores

Guardar los cambios de código/datos con commit y push, y ejecutar de nuevo
`bash scripts/deploy.sh "$PROJECT_ID"`. Cada imagen tiene un tag formado por
el commit y una marca de tiempo. Todavía no hay un workflow de GitHub Actions:
la publicación es explícita desde el equipo autenticado.

## Optimizaciones implementadas

- **Memoria:** las fechas ambientales se formatean una vez por día, en lugar
  de una vez por fila. El índice diario guarda rangos de filas y obtiene vistas
  de los datos, evitando mantener copias de las grillas por cada día.
- **Transferencia:** las respuestas de `/api/` mayores a 1.000 bytes admiten
  gzip. Las imágenes no se recomprimen al servirlas.
- **Caché HTTP:** las respuestas correctas de la API son públicas, con 5 minutos
  en navegador y 1 hora en una caché compartida compatible. Los medios se pueden
  guardar durante 24 horas. Los errores no reciben esta política.
- **Frontend:** comparte solicitudes idénticas que estén en curso; las
  posteriores respetan la caché HTTP. Un fallo de metadatos permite reintentar.
  Activar el mapa de calor ya no vuelve a consultar los datos.
- **Fotografías:** la galería usa miniaturas WebP de hasta 640 × 640, generadas
  antes del despliegue. El visor ampliado sigue solicitando el original.
  Si no existen miniaturas, la API y la galería utilizan los originales.

### Mediciones locales

Comparación del cargador de datos en procesos nuevos: Python 3.12, pandas 3.0.6,
PyArrow 26.0.0; 4.320.144 filas de temperatura y 586.872 de clorofila. El pico se
muestreó cada 5 ms. Estos números no son mediciones de una instancia Cloud Run.

| Métrica | Antes | Después |
| --- | ---: | ---: |
| Arranque del cargador, incluyendo imports | 2,07 s | 1,60 s |
| RSS al terminar la carga | 404 MiB | 324 MiB |
| Pico RSS muestreado durante la carga | 873 MiB | 490 MiB |
| JSON de `/api/sightings`, bytes transferidos | 514.678 | 40.052 con gzip |
| JSON de `/api/tracks`, bytes transferidos | 521.678 | 121.857 con gzip |
| 66 fotografías, galería completa | 147,95 MiB originales | 2,17 MiB miniaturas |

La comparación de respuestas de 10 consultas base y 48 combinaciones ambientales
dio el mismo SHA-256 antes y después de optimizar la memoria. Las miniaturas añaden
el campo opcional `thumbnail_url` a las respuestas de fotos.

El ahorro de miniaturas corresponde a visualizar la galería. Abrir originales
en el visor sigue transfiriendo esos archivos. El tiempo de arranque y la RAM
varían según CPU, versiones y tamaño de los datos; el servidor HTTP, la caché
de respuestas y las solicitudes concurrentes necesitan memoria adicional.

## Preparación y comprobaciones

Desde la raíz del repositorio:

```bash
uv run --with pillow python scripts/prepare_photos.py
uv run --with-requirements requirements-dev.txt python -m pytest -q
uv run --with-requirements requirements-dev.txt python scripts/profile_api.py
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
```

Las miniaturas se generan en `data/fotos/thumbnails/` y
`data/fotos_conteo_directo/thumbnails/`. Son artefactos ignorados por Git:
**el pipeline de despliegue debe generarlas antes de copiar los medios e iniciar
la API**. Pillow solo hace falta para esa preparación y sus pruebas.

`scripts/build_hosting.py` copia ambos directorios bajo `frontend/dist/media/`
y publica el logo en `/media/logo`; `firebase.json` define su tipo PNG. La imagen
de la API conserva también los medios para construir el catálogo y servir las
mismas URLs durante el desarrollo. Las visitas a Hosting reciben los archivos
estáticos directamente desde Firebase.

Al actualizar los Excel/Parquet/fotografías hay que regenerar las miniaturas y
reiniciar o desplegar la API para que reconstruya su catálogo. Las cachés pueden
mostrar la versión previa durante los TTL indicados. Para actualizaciones
inmediatas conviene versionar las URLs de los recursos y purgar la CDN cuando
corresponda. Los headers de archivos publicados en Hosting deben configurarse
allí: no pasan por el middleware de FastAPI.

## Configuración inicial sugerida para Cloud Run

- Facturación por solicitudes y mínimo de instancias **0**.
- Como punto de partida: **1 vCPU y 1 GiB de RAM**, con un solo worker Uvicorn.
  El pico local de 490 MiB deja muy poco margen para un límite de 512 MiB.
- Concurrencia inicial moderada, por ejemplo **8**, y máximo de **2 instancias**.
  Validar el contenedor real con varios filtros simultáneos antes de ajustar.
- Arranque: `uvicorn api.main:app --host 0.0.0.0 --port "$PORT"`.
- La imagen instala `requirements-api.lock`, generado desde `requirements-api.txt`.
  `requirements.txt` corresponde a la versión Streamlit y añade dependencias
  que esta API no necesita.
- Incluir los Excel, Parquet, GPX y medios necesarios. Excluir del contenedor
  los NetCDF de origen (aproximadamente 77 MiB), notebooks, entornos locales y
  dependencias de desarrollo. La API actual no lee esos NetCDF.
- No hace falta un volumen persistente para los datos de solo lectura: pueden
  ir en la imagen y actualizarse con cada despliegue.

## Posible siguiente etapa: Hosting sin backend

Todos los endpoints actuales son de lectura y los datos vienen de archivos.
Es viable generar los JSON/GeoJSON y las estadísticas durante el build, servir
las grillas por fecha/profundidad y mover los filtros y el CSV al navegador.
Eso permitiría publicar solo en Firebase Hosting, o incluso en una carpeta del
hosting WordPress si admite esos archivos y rutas.

Esta conversión requiere cambiar la capa de datos del frontend y automatizar
la exportación al actualizar el dataset. Conviene dividir los datos ambientales
en archivos pequeños; enviar todo el histórico a cada visitante aumentaría
transferencia y memoria del navegador. Para el volumen previsto, se puede
empezar con Hosting + Cloud Run y evaluar esa conversión según el costo real.

## Comprobación local de la infraestructura

```bash
docker build -t avistamientos-api:local .
docker run --rm --memory 1g --cpus 1 -p 127.0.0.1:8080:8080 avistamientos-api:local
```

En otra terminal se puede consultar `http://127.0.0.1:8080/api/meta`.
Para revisar los estáticos y rutas SPA:

```bash
uv run --python 3.12 --with pillow==13.0.0 python scripts/build_hosting.py
npx --yes firebase-tools@15.34.0 emulators:start --only hosting --project demo-avistamientos
```

El emulador de Hosting no sustituye al servicio Cloud Run del rewrite. La
integración `/api/**` debe comprobarse con el proyecto real después del primer
despliegue. Se verificaron localmente el contenedor, ocho consultas API
simultáneas con el límite de 1 GiB, las rutas SPA, el logo, las miniaturas y los
headers de caché. Los recursos reales de Firebase/GCP se crean al ejecutar los
pasos de preparación/publicación con el Project ID real.
