import pandas as pd

# Ruta al archivo CSV
archivo_csv = 'Preprocessed_dataset.csv'

# Cargar el dataset en un DataFrame
df = pd.read_csv(archivo_csv)

# Ver las primeras filas del DataFrame para verificar su correcta carga
print(df.head())

# Definir el número máximo de palabras permitidas por entrada
max_words = 200

# Función para truncar las entradas de texto a un número máximo de palabras
def truncate_text(text, limit=max_words):
    words = text.split()  # Dividir el texto en palabras
    if len(words) > limit:
        return ' '.join(words[:limit])  # Unir las primeras 'limit' palabras
    else:
        return text

# Asumiendo que la columna que quieres truncar se llama 'texto'
df['contexts'] = df['contexts'].apply(truncate_text)

# Ver las primeras filas del DataFrame modificado
print(df.head())

# Ruta al archivo CSV de salida
archivo_csv_salida = 'phishing_dataset.csv'

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv(archivo_csv_salida, index=False)

print("Dataset modificado guardado con éxito.")
