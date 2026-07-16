using MPIPreferences
using Libdl

# 1. Buscar el ejecutable mpiexec en el PATH
const MPIEXEC = strip(String(read(`which mpiexec`)))

# 2. Buscar las librerías en el LD_LIBRARY_PATH
# Tu .bashrc ya carga module load openmpi/4.1.6, así que las rutas están aquí:
paths = split(ENV["LD_LIBRARY_PATH"], ':')

function find_lib(name_pattern, search_paths)
    for path in search_paths
        if isdir(path)
            # Buscamos archivos que empiecen por el nombre y sean .so
            files = readdir(path)
            # Filtro: empieza por el nombre, contiene .so, y no es un symlink roto
            candidates = filter(f -> startswith(f, name_pattern) && contains(f, ".so"), files)
            if !isempty(candidates)
                # Devolvemos la ruta completa del primero que encontremos
                full_path = joinpath(path, first(candidates))
                return full_path
            end
        end
    end
    return nothing
end

# Buscamos la de C (libmpi) y la de Fortran (libmpi_mpifh)
libmpi_c = find_lib("libmpi.so", paths)
libmpi_f = find_lib("libmpi_mpifh.so", paths)

# Verificación
if isnothing(libmpi_c) || isnothing(libmpi_f)
    println("Error: No se encontraron las librerías en LD_LIBRARY_PATH.")
    println("Rutas buscadas: ", paths)
    exit(1)
end

println("=== Configurando MPIPreferences ===")
println("Ejecutable: $MPIEXEC")
println("Lib C:      $libmpi_c")
println("Lib F:      $libmpi_f")

# 3. Aplicar la configuración forzando AMBAS librerías
MPIPreferences.use_system_binary(;
    library_names = [libmpi_c, libmpi_f], # <--- ¡Aquí está la clave!
    mpiexec = MPIEXEC,
    export_prefs = true,
    force = true
)

println("=== ¡Configuración Exitosa! ===")
println("Se ha generado el archivo LocalPreferences.toml en la carpeta superior.")