#!/usr/bin/env python3
"""
Sistema de Análisis de Satélites
NASA Space App Challenge 2025

Este sistema permite:
1. Obtener datos de satélites desde Celestrak usando Skyfield
2. Buscar satélites por nombre
3. Calcular órbitas y posiciones futuras
4. Predecir posibles colisiones en los próximos 6 meses
5. Visualizar trayectorias orbitales

Autor: NASA Space App Team
Fecha: Octubre 2025
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from skyfield.api import load, EarthSatellite
from skyfield.timelib import Time
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importaciones para visualización 3D
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SatelliteAnalyzer:
    """
    Clase principal para análisis de satélites usando Skyfield y datos de Celestrak
    """
    
    def __init__(self):
        """Inicializar el analizador de satélites"""
        self.ts = load.timescale()
        self.satellites = {}
        self.tle_data = {}
        self.earth = load('de421.bsp')['earth']
        print("🛰️  Inicializando Sistema de Análisis de Satélites...")
        
    def download_tle_data(self, tle_url: str = None) -> bool:
        """
        Descargar datos TLE (Two-Line Elements) desde Celestrak
        
        Args:
            tle_url: URL personalizada para datos TLE
            
        Returns:
            bool: True si la descarga fue exitosa
        """
        try:
            # URLs de diferentes categorías de satélites de Celestrak
            urls = {
                'active': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
                'stations': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle',
                'weather': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle',
                'communications': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
                'navigation': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle'
            }
            
            if tle_url:
                urls['custom'] = tle_url
                
            print("📡 Descargando datos TLE desde Celestrak...")
            
            all_satellites = {}
            for category, url in urls.items():
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Parsear datos TLE
                    lines = response.text.strip().split('\n')
                    i = 0
                    while i < len(lines) - 2:
                        if lines[i].strip() and not lines[i].startswith('#'):
                            name = lines[i].strip()
                            line1 = lines[i + 1].strip()
                            line2 = lines[i + 2].strip()
                            
                            if line1.startswith('1 ') and line2.startswith('2 '):
                                # Crear satélite usando Skyfield
                                satellite = EarthSatellite(line1, line2, name, self.ts)
                                all_satellites[name] = {
                                    'satellite': satellite,
                                    'line1': line1,
                                    'line2': line2,
                                    'category': category
                                }
                        i += 3
                        
                    print(f"   ✅ {category}: {len([s for s in all_satellites.values() if s['category'] == category])} satélites")
                    
                except Exception as e:
                    print(f"   ❌ Error descargando {category}: {str(e)}")
                    continue
            
            self.satellites = all_satellites
            print(f"🎯 Total de satélites cargados: {len(self.satellites)}")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando datos TLE: {str(e)}")
            return False
    
    def export_satellites_list(self, filename: str = "satelites_disponibles.txt") -> bool:
        """
        Exportar lista de todos los satélites disponibles a un archivo de texto
        
        Args:
            filename: Nombre del archivo a crear
            
        Returns:
            bool: True si la exportación fue exitosa
        """
        try:
            if not self.satellites:
                print("❌ No hay satélites cargados. Ejecuta download_tle_data() primero.")
                return False
            
            # Organizar satélites por categoría
            satellites_by_category = {}
            for name, data in self.satellites.items():
                category = data['category']
                if category not in satellites_by_category:
                    satellites_by_category[category] = []
                satellites_by_category[category].append(name)
            
            # Ordenar satélites alfabéticamente dentro de cada categoría
            for category in satellites_by_category:
                satellites_by_category[category].sort()
            
            # Crear el archivo de texto
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LISTA DE SATÉLITES DISPONIBLES\n")
                f.write("Sistema de Análisis de Satélites - NASA Space App Challenge 2025\n")
                f.write("=" * 80 + "\n")
                f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de satélites: {len(self.satellites)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Escribir resumen por categoría
                f.write("RESUMEN POR CATEGORÍA:\n")
                f.write("-" * 40 + "\n")
                total_count = 0
                for category, sat_list in satellites_by_category.items():
                    count = len(sat_list)
                    total_count += count
                    f.write(f"{category.capitalize():20s}: {count:5d} satélites\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'TOTAL':20s}: {total_count:5d} satélites\n\n")
                
                # Escribir lista detallada por categoría
                for category, sat_list in satellites_by_category.items():
                    f.write("=" * 80 + "\n")
                    f.write(f"CATEGORÍA: {category.upper()}\n")
                    f.write(f"Total en esta categoría: {len(sat_list)} satélites\n")
                    f.write("=" * 80 + "\n")
                    
                    for i, sat_name in enumerate(sat_list, 1):
                        f.write(f"{i:4d}. {sat_name}\n")
                    
                    f.write("\n")
                
                # Agregar lista alfabética completa
                f.write("=" * 80 + "\n")
                f.write("LISTA ALFABÉTICA COMPLETA\n")
                f.write("=" * 80 + "\n")
                
                all_satellites = sorted(self.satellites.keys())
                for i, sat_name in enumerate(all_satellites, 1):
                    category = self.satellites[sat_name]['category']
                    f.write(f"{i:5d}. {sat_name:<50s} [{category}]\n")
                
                # Agregar información útil al final
                f.write("\n" + "=" * 80 + "\n")
                f.write("INSTRUCCIONES DE USO:\n")
                f.write("=" * 80 + "\n")
                f.write("1. Copia el nombre exacto del satélite que deseas analizar\n")
                f.write("2. Pégalo en el programa cuando se solicite el nombre\n")
                f.write("3. Los nombres son sensibles a mayúsculas y minúsculas\n")
                f.write("4. Usa Ctrl+F para buscar satélites específicos en este archivo\n\n")
                
                f.write("EJEMPLOS DE SATÉLITES INTERESANTES:\n")
                f.write("-" * 40 + "\n")
                
                # Buscar algunos satélites interesantes como ejemplos
                interesting_examples = []
                search_terms = ["ISS", "HUBBLE", "NOAA", "GPS", "STARLINK", "GOES"]
                
                for term in search_terms:
                    matches = [name for name in all_satellites if term in name.upper()]
                    if matches:
                        interesting_examples.append(f"• {matches[0]} (búsqueda: '{term}')")
                
                for example in interesting_examples:
                    f.write(f"{example}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("¡Explora el cosmos, un satélite a la vez! 🛰️🌌\n")
                f.write("=" * 80 + "\n")
            
            print(f"✅ Lista de satélites exportada exitosamente:")
            print(f"   📁 Archivo: {filename}")
            print(f"   🛰️  Total de satélites: {len(self.satellites)}")
            print(f"   📂 Categorías: {len(satellites_by_category)}")
            return True
            
        except Exception as e:
            print(f"❌ Error exportando lista de satélites: {str(e)}")
            return False
    
    def search_satellite(self, search_term: str) -> List[str]:
        """
        Buscar satélites por nombre
        
        Args:
            search_term: Término de búsqueda
            
        Returns:
            List[str]: Lista de nombres de satélites que coinciden
        """
        search_term = search_term.lower()
        matches = []
        
        for name in self.satellites.keys():
            if search_term in name.lower():
                matches.append(name)
                
        return sorted(matches)
    
    def get_popular_satellites(self) -> Dict[str, List[str]]:
        """
        Obtener una lista de satélites populares organizados por categoría
        
        Returns:
            Dict: Diccionario con categorías y satélites populares
        """
        popular_categories = {
            'Estaciones Espaciales': ['ISS', 'ZARYA', 'TIANGONG'],
            'Telescopios Espaciales': ['HUBBLE', 'SPITZER', 'CHANDRA'],
            'Satélites Meteorológicos': ['NOAA', 'GOES', 'METEOSAT'],
            'Navegación GPS': ['GPS', 'NAVSTAR', 'GLONASS'],
            'Comunicaciones': ['STARLINK', 'INTELSAT', 'IRIDIUM'],
            'Observación Terrestre': ['LANDSAT', 'AQUA', 'TERRA', 'SENTINEL']
        }
        
        found_satellites = {}
        
        for category, search_terms in popular_categories.items():
            found_satellites[category] = []
            for term in search_terms:
                matches = self.search_satellite(term)
                if matches:
                    # Agregar los primeros 3 matches de cada término
                    found_satellites[category].extend(matches[:3])
            
            # Remover duplicados y limitar a 5 por categoría
            found_satellites[category] = list(dict.fromkeys(found_satellites[category]))[:5]
        
        return found_satellites
    
    def suggest_satellites(self, partial_name: str) -> List[str]:
        """
        Sugerir satélites basándose en un nombre parcial
        
        Args:
            partial_name: Nombre parcial del satélite
            
        Returns:
            List[str]: Lista de sugerencias
        """
        if len(partial_name) < 2:
            return []
        
        partial_name = partial_name.lower()
        suggestions = []
        
        # Buscar coincidencias exactas al inicio del nombre
        for name in self.satellites.keys():
            if name.lower().startswith(partial_name):
                suggestions.append(name)
        
        # Si no hay suficientes, buscar coincidencias en cualquier parte
        if len(suggestions) < 10:
            for name in self.satellites.keys():
                if partial_name in name.lower() and name not in suggestions:
                    suggestions.append(name)
        
        return sorted(suggestions)[:15]  # Limitar a 15 sugerencias
    
    def browse_satellites_by_category(self) -> Dict[str, List[str]]:
        """
        Navegar satélites organizados por categoría
        
        Returns:
            Dict: Satélites organizados por categoría con muestras
        """
        satellites_by_category = {}
        
        for name, data in self.satellites.items():
            category = data['category']
            if category not in satellites_by_category:
                satellites_by_category[category] = []
            satellites_by_category[category].append(name)
        
        # Ordenar y limitar para navegación fácil
        for category in satellites_by_category:
            satellites_by_category[category] = sorted(satellites_by_category[category])
        
        return satellites_by_category
    
    def show_satellite_examples(self) -> None:
        """
        Mostrar ejemplos de satélites interesantes con descripción
        """
        examples = {
            "🏠 Estaciones Espaciales": {
                "search_terms": ["ISS", "ZARYA", "TIANGONG"],
                "description": "Laboratorios orbitales tripulados"
            },
            "🔭 Telescopios Espaciales": {
                "search_terms": ["HUBBLE", "SPITZER", "KEPLER"],
                "description": "Observatorios astronómicos en el espacio"
            },
            "🌤️ Satélites Meteorológicos": {
                "search_terms": ["NOAA", "GOES", "METEOSAT"],
                "description": "Monitoreo del clima y tiempo"
            },
            "🗺️ Navegación GPS": {
                "search_terms": ["GPS", "NAVSTAR", "GALILEO"],
                "description": "Sistemas de posicionamiento global"
            },
            "📡 Comunicaciones": {
                "search_terms": ["STARLINK", "IRIDIUM", "INTELSAT"],
                "description": "Internet y telecomunicaciones"
            },
            "🌍 Observación Terrestre": {
                "search_terms": ["LANDSAT", "AQUA", "TERRA"],
                "description": "Monitoreo ambiental y recursos"
            }
        }
        
        print("\n🌟 EJEMPLOS DE SATÉLITES INTERESANTES:")
        print("=" * 60)
        
        for category, info in examples.items():
            print(f"\n{category}")
            print(f"📝 {info['description']}")
            found_examples = []
            
            for term in info['search_terms']:
                matches = self.search_satellite(term)
                if matches:
                    found_examples.extend(matches[:2])  # Máximo 2 por término
            
            # Mostrar ejemplos únicos
            unique_examples = list(dict.fromkeys(found_examples))[:3]
            for i, example in enumerate(unique_examples, 1):
                print(f"   {i}. {example}")
            
            if not unique_examples:
                print("   (No se encontraron ejemplos en los datos actuales)")
        
        print(f"\n💡 TIP: Usa la opción 1 para buscar cualquiera de estos nombres")
        print(f"🔍 Ejemplo: busca 'ISS' para encontrar la Estación Espacial Internacional")
    
    def smart_search(self, search_term: str) -> Dict:
        """
        Búsqueda inteligente que proporciona resultados y sugerencias
        
        Args:
            search_term: Término de búsqueda
            
        Returns:
            Dict: Resultados detallados con sugerencias
        """
        results = {
            'exact_matches': [],
            'partial_matches': [],
            'suggestions': [],
            'category_matches': {},
            'total_found': 0
        }
        
        if not search_term or len(search_term) < 2:
            return results
        
        search_lower = search_term.lower()
        
        # Buscar coincidencias exactas
        for name in self.satellites.keys():
            name_lower = name.lower()
            if search_lower == name_lower:
                results['exact_matches'].append(name)
            elif search_lower in name_lower:
                results['partial_matches'].append(name)
        
        # Organizar por categoría
        for name in results['partial_matches']:
            category = self.satellites[name]['category']
            if category not in results['category_matches']:
                results['category_matches'][category] = []
            results['category_matches'][category].append(name)
        
        # Generar sugerencias si hay pocos resultados
        if len(results['partial_matches']) < 10:
            results['suggestions'] = self.suggest_satellites(search_term)
        
        results['total_found'] = len(results['exact_matches']) + len(results['partial_matches'])
        
        return results
    
    def get_satellite_info(self, satellite_name: str) -> Optional[Dict]:
        """
        Obtener información detallada de un satélite
        
        Args:
            satellite_name: Nombre del satélite
            
        Returns:
            Dict: Información del satélite o None si no se encuentra
        """
        if satellite_name not in self.satellites:
            return None
            
        sat_data = self.satellites[satellite_name]
        satellite = sat_data['satellite']
        
        # Tiempo actual
        now = self.ts.now()
        
        # Calcular posición actual
        geocentric = satellite.at(now)
        subpoint = geocentric.subpoint()
        
        # Extraer elementos orbitales del TLE
        line1 = sat_data['line1']
        line2 = sat_data['line2']
        
        # Parsear elementos orbitales
        inclination = float(line2[8:16])
        raan = float(line2[17:25])  # Right Ascension of Ascending Node
        eccentricity = float('0.' + line2[26:33])
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        
        # Calcular período orbital
        period_minutes = 1440 / mean_motion  # minutos
        period_hours = period_minutes / 60
        
        # Calcular altitud aproximada
        # Usando la tercera ley de Kepler: n = sqrt(GM/a³)
        GM = 398600.4418  # km³/s²
        n_rad_per_sec = mean_motion * 2 * np.pi / 86400  # radianes por segundo
        semi_major_axis = (GM / (n_rad_per_sec ** 2)) ** (1/3)
        
        altitude_km = semi_major_axis - 6371  # Radio terrestre aprox
        
        info = {
            'name': satellite_name,
            'category': sat_data['category'],
            'current_position': {
                'latitude': subpoint.latitude.degrees,
                'longitude': subpoint.longitude.degrees,
                'altitude_km': subpoint.elevation.km
            },
            'orbital_elements': {
                'inclination_deg': inclination,
                'raan_deg': raan,
                'eccentricity': eccentricity,
                'argument_of_perigee_deg': arg_perigee,
                'mean_anomaly_deg': mean_anomaly,
                'mean_motion_rev_per_day': mean_motion,
                'period_hours': period_hours,
                'semi_major_axis_km': semi_major_axis,
                'approx_altitude_km': altitude_km
            },
            'tle_data': {
                'line1': line1,
                'line2': line2
            }
        }
        
        return info
    
    def calculate_future_positions(self, satellite_name: str, days_ahead: int = 180) -> List[Dict]:
        """
        Calcular posiciones futuras del satélite
        
        Args:
            satellite_name: Nombre del satélite
            days_ahead: Días hacia el futuro para calcular
            
        Returns:
            List[Dict]: Posiciones futuras del satélite
        """
        if satellite_name not in self.satellites:
            return []
            
        satellite = self.satellites[satellite_name]['satellite']
        
        # Crear timestamps para los próximos días
        start_time = self.ts.now()
        positions = []
        
        # Calcular posiciones cada 12 horas
        for hours in range(0, days_ahead * 24, 12):
            t = self.ts.tt_jd(start_time.tt + hours / 24)
            geocentric = satellite.at(t)
            subpoint = geocentric.subpoint()
            
            positions.append({
                'datetime': t.utc_datetime(),
                'latitude': subpoint.latitude.degrees,
                'longitude': subpoint.longitude.degrees,
                'altitude_km': subpoint.elevation.km,
                'x_km': geocentric.position.km[0],
                'y_km': geocentric.position.km[1],
                'z_km': geocentric.position.km[2]
            })
            
        return positions
    
    def analyze_collision_risk(self, satellite1_name: str, satellite2_name: str = None, 
                             threshold_km: float = 10.0, days_ahead: int = 180) -> Dict:
        """
        Analizar riesgo de colisión entre satélites
        
        Args:
            satellite1_name: Primer satélite
            satellite2_name: Segundo satélite (si None, analiza contra todos)
            threshold_km: Distancia mínima para considerar riesgo
            days_ahead: Días a analizar hacia el futuro
            
        Returns:
            Dict: Análisis de riesgo de colisión
        """
        if satellite1_name not in self.satellites:
            return {'error': f'Satélite {satellite1_name} no encontrado'}
            
        sat1 = self.satellites[satellite1_name]['satellite']
        close_encounters = []
        
        # Determinar satélites a analizar
        satellites_to_check = {}
        if satellite2_name:
            if satellite2_name in self.satellites:
                satellites_to_check[satellite2_name] = self.satellites[satellite2_name]
        else:
            # Analizar contra una muestra de satélites (primeros 100 para eficiencia)
            sat_names = list(self.satellites.keys())[:100]
            for name in sat_names:
                if name != satellite1_name:
                    satellites_to_check[name] = self.satellites[name]
        
        print(f"🔍 Analizando {len(satellites_to_check)} satélites para posibles colisiones...")
        
        # Analizar cada 6 horas durante el período especificado
        for hours in range(0, days_ahead * 24, 6):
            t = self.ts.tt_jd(self.ts.now().tt + hours / 24)
            pos1 = sat1.at(t)
            
            for sat2_name, sat2_data in satellites_to_check.items():
                sat2 = sat2_data['satellite']
                pos2 = sat2.at(t)
                
                # Calcular distancia entre satélites
                distance_km = np.linalg.norm(
                    np.array(pos1.position.km) - np.array(pos2.position.km)
                )
                
                if distance_km < threshold_km:
                    close_encounters.append({
                        'datetime': t.utc_datetime(),
                        'satellite2': sat2_name,
                        'distance_km': distance_km,
                        'satellite1_pos': pos1.position.km,
                        'satellite2_pos': pos2.position.km
                    })
        
        # Calcular estadísticas de riesgo
        risk_level = 'BAJO'
        if close_encounters:
            min_distance = min(enc['distance_km'] for enc in close_encounters)
            if min_distance < 1.0:
                risk_level = 'CRÍTICO'
            elif min_distance < 5.0:
                risk_level = 'ALTO'
            else:
                risk_level = 'MEDIO'
        
        return {
            'satellite': satellite1_name,
            'analysis_period_days': days_ahead,
            'threshold_km': threshold_km,
            'close_encounters': close_encounters,
            'risk_level': risk_level,
            'total_encounters': len(close_encounters),
            'satellites_analyzed': len(satellites_to_check)
        }
    
    def plot_orbit(self, satellite_name: str, hours: int = 24) -> bool:
        """
        Visualizar la órbita de un satélite
        
        Args:
            satellite_name: Nombre del satélite
            hours: Horas de órbita a mostrar
            
        Returns:
            bool: True si el plot fue exitoso
        """
        if satellite_name not in self.satellites:
            print(f"❌ Satélite {satellite_name} no encontrado")
            return False
            
        satellite = self.satellites[satellite_name]['satellite']
        
        # Calcular posiciones para la visualización
        positions = []
        times = []
        
        start_time = self.ts.now()
        for minutes in range(0, hours * 60, 10):  # Cada 10 minutos
            t = self.ts.tt_jd(start_time.tt + minutes / (24 * 60))
            geocentric = satellite.at(t)
            subpoint = geocentric.subpoint()
            
            positions.append([
                subpoint.longitude.degrees,
                subpoint.latitude.degrees
            ])
            times.append(t.utc_datetime())
        
        positions = np.array(positions)
        
        # Crear el plot
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Trayectoria en mapa mundial
        plt.subplot(1, 2, 1)
        plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
        plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, 
                   label='Inicio', zorder=5)
        plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, 
                   label='Fin', zorder=5)
        
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.xlabel('Longitud (°)')
        plt.ylabel('Latitud (°)')
        plt.title(f'Trayectoria Orbital: {satellite_name}\n({hours} horas)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Agregar líneas de referencia
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Subplot 2: Altitud vs tiempo
        plt.subplot(1, 2, 2)
        altitudes = []
        for minutes in range(0, hours * 60, 10):
            t = self.ts.tt_jd(start_time.tt + minutes / (24 * 60))
            geocentric = satellite.at(t)
            subpoint = geocentric.subpoint()
            altitudes.append(subpoint.elevation.km)
        
        time_hours = [i/6 for i in range(len(altitudes))]  # Cada 10 min = 1/6 hora
        plt.plot(time_hours, altitudes, 'r-', linewidth=2)
        plt.xlabel('Tiempo (horas)')
        plt.ylabel('Altitud (km)')
        plt.title('Variación de Altitud')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar el plot
        filename = f"orbit_{satellite_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado como: {filename}")
        
        plt.show()
        return True
    
    def plot_3d_earth_with_satellites(self, satellite_names: List[str], hours: int = 24) -> bool:
        """
        Visualización 3D de la Tierra con trayectorias de satélites
        
        Args:
            satellite_names: Lista de nombres de satélites a visualizar
            hours: Horas de órbita a mostrar
            
        Returns:
            bool: True si la visualización fue exitosa
        """
        if not satellite_names:
            print("❌ No se proporcionaron nombres de satélites")
            return False
        
        # Verificar que los satélites existen
        valid_satellites = []
        for name in satellite_names:
            if name in self.satellites:
                valid_satellites.append(name)
            else:
                print(f"⚠️  Satélite {name} no encontrado")
        
        if not valid_satellites:
            print("❌ No se encontraron satélites válidos")
            return False
        
        print(f"🌍 Generando visualización 3D para {len(valid_satellites)} satélite(s)...")
        
        # Crear figura de Plotly
        fig = go.Figure()
        
        # Agregar la Tierra como una esfera
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = 6371 * np.outer(np.cos(u), np.sin(v))  # Radio de la Tierra: 6371 km
        y_earth = 6371 * np.outer(np.sin(u), np.sin(v))
        z_earth = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            opacity=0.7,
            name='Tierra',
            showscale=False,
            hovertemplate='Tierra<extra></extra>'
        ))
        
        # Colores para diferentes satélites
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan']
        
        # Agregar trayectorias de satélites
        for i, satellite_name in enumerate(valid_satellites):
            satellite = self.satellites[satellite_name]['satellite']
            color = colors[i % len(colors)]
            
            # Calcular posiciones del satélite
            positions_3d = []
            times = []
            
            start_time = self.ts.now()
            for minutes in range(0, hours * 60, 15):  # Cada 15 minutos para mejor rendimiento
                t = self.ts.tt_jd(start_time.tt + minutes / (24 * 60))
                geocentric = satellite.at(t)
                
                # Convertir a coordenadas cartesianas (km)
                position = geocentric.position.km
                positions_3d.append(position)
                times.append(t.utc_datetime())
            
            positions_3d = np.array(positions_3d)
            
            # Agregar trayectoria del satélite
            fig.add_trace(go.Scatter3d(
                x=positions_3d[:, 0],
                y=positions_3d[:, 1], 
                z=positions_3d[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color),
                name=f'{satellite_name}',
                hovertemplate=f'<b>{satellite_name}</b><br>' +
                            'X: %{x:.1f} km<br>' +
                            'Y: %{y:.1f} km<br>' +
                            'Z: %{z:.1f} km<extra></extra>'
            ))
            
            # Marcar posición inicial y final
            fig.add_trace(go.Scatter3d(
                x=[positions_3d[0, 0]],
                y=[positions_3d[0, 1]],
                z=[positions_3d[0, 2]],
                mode='markers',
                marker=dict(size=8, color='lightgreen', symbol='diamond'),
                name=f'{satellite_name} - Inicio',
                showlegend=False,
                hovertemplate=f'<b>{satellite_name} - Inicio</b><br>' +
                            'X: %{x:.1f} km<br>' +
                            'Y: %{y:.1f} km<br>' +
                            'Z: %{z:.1f} km<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[positions_3d[-1, 0]],
                y=[positions_3d[-1, 1]],
                z=[positions_3d[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='darkred', symbol='cross'),
                name=f'{satellite_name} - Final',
                showlegend=False,
                hovertemplate=f'<b>{satellite_name} - Final</b><br>' +
                            'X: %{x:.1f} km<br>' +
                            'Y: %{y:.1f} km<br>' +
                            'Z: %{z:.1f} km<extra></extra>'
            ))
        
        # Configurar el layout
        fig.update_layout(
            title=f'🛰️ Visualización 3D: Satélites alrededor de la Tierra<br>' +
                  f'<sub>Trayectorias de {hours} horas - {len(valid_satellites)} satélite(s)</sub>',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2, y=2, z=2)
                ),
                bgcolor='black'
            ),
            font=dict(size=12),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Mostrar la visualización
        fig.show()
        
        # Guardar como HTML interactivo
        filename = f"satellite_3d_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"🌍 Visualización 3D guardada como: {filename}")
        
        return True
    
    def plot_orbital_animation(self, satellite_name: str, hours: int = 24, frames: int = 100) -> bool:
        """
        Crear una animación de la órbita del satélite alrededor de la Tierra
        
        Args:
            satellite_name: Nombre del satélite
            hours: Horas de órbita a animar
            frames: Número de frames en la animación
            
        Returns:
            bool: True si la animación fue exitosa
        """
        try:
            if satellite_name not in self.satellites:
                print(f"❌ Satélite {satellite_name} no encontrado")
                return False
            
            satellite = self.satellites[satellite_name]['satellite']
            print(f"🎬 Generando animación orbital para {satellite_name}...")
            print(f"⏱️  Calculando {frames} posiciones para {hours} horas...")
            
            # Calcular todas las posiciones
            all_positions = []
            start_time = self.ts.now()
            
            for frame in range(frames + 1):
                minutes = (hours * 60 * frame) / frames
                t = self.ts.tt_jd(start_time.tt + minutes / (24 * 60))
                geocentric = satellite.at(t)
                position = geocentric.position.km
                all_positions.append(position)
            
            all_positions = np.array(all_positions)
            print(f"✅ Posiciones calculadas")
            
            # Crear la animación
            fig = go.Figure()
            
            # Agregar la Tierra con un colorscale más simple
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_earth = 6371 * np.outer(np.cos(u), np.sin(v))
            y_earth = 6371 * np.outer(np.sin(u), np.sin(v))
            z_earth = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x_earth, y=y_earth, z=z_earth,
                colorscale='Blues',  # Cambié de 'Earth' a 'Blues' para mayor compatibilidad
                opacity=0.8,
                name='Tierra',
                showscale=False,
                hovertemplate='Tierra<extra></extra>'
            ))
            
            print(f"🌍 Tierra agregada a la visualización")
            
            # Crear frames para la animación (reducir cantidad para mejor rendimiento)
            frames_list = []
            step = max(1, frames // 20)  # Máximo 20 frames para mejor rendimiento
            
            for i in range(0, frames + 1, step):
                if i >= len(all_positions):
                    break
                    
                frame_data = [
                    go.Surface(
                        x=x_earth, y=y_earth, z=z_earth,
                        colorscale='Blues',
                        opacity=0.8,
                        showscale=False,
                        hovertemplate='Tierra<extra></extra>'
                    ),
                    go.Scatter3d(
                        x=all_positions[:i+1, 0],
                        y=all_positions[:i+1, 1],
                        z=all_positions[:i+1, 2],
                        mode='lines',
                        line=dict(color='red', width=6),
                        name='Trayectoria',
                        hovertemplate='Trayectoria<extra></extra>'
                    ),
                    go.Scatter3d(
                        x=[all_positions[i, 0]],
                        y=[all_positions[i, 1]],
                        z=[all_positions[i, 2]],
                        mode='markers',
                        marker=dict(size=12, color='yellow', symbol='circle'),
                        name='Satélite',
                        hovertemplate=f'{satellite_name}<br>X: %{{x:.1f}} km<br>Y: %{{y:.1f}} km<br>Z: %{{z:.1f}} km<extra></extra>'
                    )
                ]
                frames_list.append(go.Frame(data=frame_data, name=str(i)))
            
            fig.frames = frames_list
            print(f"🎞️  {len(frames_list)} frames de animación creados")
            
            # Configurar la animación con controles mejorados
            fig.update_layout(
                title=f'🎬 Animación Orbital: {satellite_name}<br><sub>Período: {hours} horas | Frames: {len(frames_list)}</sub>',
                scene=dict(
                    xaxis_title='X (km)',
                    yaxis_title='Y (km)', 
                    zaxis_title='Z (km)',
                    aspectmode='cube',
                    camera=dict(eye=dict(x=2.5, y=2.5, z=2.5)),
                    bgcolor='black'
                ),
                font=dict(size=12),
                width=1000,
                height=700,
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.1,
                    'y': 0.02,
                    'buttons': [
                        {
                            'label': '▶️ Reproducir',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 200, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 100}
                            }]
                        },
                        {
                            'label': '⏸️ Pausar',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        },
                        {
                            'label': '🔄 Reiniciar',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 200, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 20},
                        'prefix': 'Frame:',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'transition': {'duration': 100, 'easing': 'cubic-in-out'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [[f.name], {
                                'frame': {'duration': 100, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 100}
                            }],
                            'label': f.name,
                            'method': 'animate'
                        } for f in frames_list
                    ]
                }]
            )
            
            print(f"🎨 Configuración de animación completada")
            
            # Mostrar la visualización
            print(f"🌐 Abriendo animación en el navegador...")
            fig.show()
            
            # Guardar como HTML
            safe_name = satellite_name.replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"animacion_orbital_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(filename)
            print(f"💾 Animación guardada como: {filename}")
            print(f"📁 Ubicación: {os.path.abspath(filename)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creando animación: {str(e)}")
            print(f"💡 Sugerencias:")
            print(f"   1. Verifica que el nombre del satélite sea correcto")
            print(f"   2. Intenta con menos frames (ej: 20-30)")
            print(f"   3. Reduce las horas (ej: 2-6 horas)")
            return False


def mostrar_menu():
    """Mostrar el menú de opciones"""
    print("\n" + "=" * 60)
    print("🎯 OPCIONES DISPONIBLES:")
    print("   1. Buscar satélite (búsqueda inteligente)")
    print("   2. Ver satélites populares por categoría")
    print("   3. Información detallada de un satélite")
    print("   4. Calcular órbitas futuras")
    print("   5. Análizar riesgo de colisión")
    print("   6. Visualizar órbita (2D)")
    print("   7. Visualización 3D (Tierra + Satélites)")
    print("   8. Animación orbital 3D")
    print("   9. Exportar lista completa de satélites")
    print("  10. Salir")
    print("=" * 60)


def main():
    """Función principal del programa"""
    print("=" * 60)
    print("🛰️  SISTEMA DE ANÁLISIS DE SATÉLITES")
    print("    NASA Space App Challenge 2025")
    print("=" * 60)
    
    # Inicializar el analizador
    analyzer = SatelliteAnalyzer()
    
    # Descargar datos de satélites
    if not analyzer.download_tle_data():
        print("❌ Error al descargar datos. Terminando programa.")
        return
    
    print("\n� ¡Sistema cargado exitosamente!")
    
    # El menú se mostrará automáticamente en cada iteración del bucle
    
    while True:
        try:
            # Mostrar menú en cada iteración
            mostrar_menu()
            print("\n" + "-" * 40)
            option = input("Selecciona una opción (1-10): ").strip()
            
            if option == '1':
                # Búsqueda inteligente de satélite
                search_term = input("🔍 Ingresa el nombre del satélite a buscar: ").strip()
                if search_term:
                    results = analyzer.smart_search(search_term)
                    
                    if results['total_found'] > 0:
                        print(f"\n✅ Encontrados {results['total_found']} satélites:")
                        
                        # Mostrar coincidencias exactas primero
                        if results['exact_matches']:
                            print("\n🎯 COINCIDENCIAS EXACTAS:")
                            for i, name in enumerate(results['exact_matches'], 1):
                                print(f"   {i}. {name}")
                        
                        # Mostrar coincidencias parciales por categoría
                        if results['category_matches']:
                            print("\n📊 RESULTADOS POR CATEGORÍA:")
                            for category, satellites in results['category_matches'].items():
                                print(f"\n   📂 {category.capitalize()}:")
                                for i, name in enumerate(satellites[:5], 1):  # Máximo 5 por categoría
                                    print(f"      {i}. {name}")
                                if len(satellites) > 5:
                                    print(f"      ... y {len(satellites) - 5} más en esta categoría")
                        
                        # Mostrar sugerencias si hay pocas coincidencias
                        if results['suggestions'] and results['total_found'] < 5:
                            print(f"\n💡 SUGERENCIAS RELACIONADAS:")
                            for i, suggestion in enumerate(results['suggestions'][:8], 1):
                                print(f"   {i}. {suggestion}")
                    else:
                        print("❌ No se encontraron satélites con ese nombre")
                        
                        # Mostrar ejemplos populares
                        print("\n🌟 ¿Quizás buscabas alguno de estos satélites populares?")
                        analyzer.show_satellite_examples()
                        
            elif option == '2':
                # Ver satélites populares por categoría
                print("🌟 SATÉLITES POPULARES POR CATEGORÍA")
                print("=" * 50)
                
                popular = analyzer.get_popular_satellites()
                
                for category, satellites in popular.items():
                    if satellites:
                        print(f"\n📂 {category}:")
                        for i, sat in enumerate(satellites, 1):
                            print(f"   {i}. {sat}")
                    else:
                        print(f"\n📂 {category}: (No se encontraron en los datos actuales)")
                
                print(f"\n💡 TIP: Copia cualquier nombre exacto para usarlo en otras opciones")
                
            elif option == '3':
                # Información detallada
                sat_name = input("📡 Ingresa el nombre exacto del satélite: ").strip()
                if not sat_name:
                    print("❌ Nombre vacío")
                elif sat_name not in analyzer.satellites:
                    print(f"❌ Satélite '{sat_name}' no encontrado")
                    # Ofrecer sugerencias
                    suggestions = analyzer.suggest_satellites(sat_name)
                    if suggestions:
                        print(f"\n🔍 ¿Quisiste decir alguno de estos?")
                        for i, suggestion in enumerate(suggestions[:5], 1):
                            print(f"   {i}. {suggestion}")
                else:
                    info = analyzer.get_satellite_info(sat_name)
                    if info:
                        print(f"\n📊 INFORMACIÓN DE {info['name']}")
                        print("-" * 50)
                        print(f"Categoría: {info['category']}")
                        print(f"Posición actual:")
                        print(f"  • Latitud: {info['current_position']['latitude']:.4f}°")
                        print(f"  • Longitud: {info['current_position']['longitude']:.4f}°")
                        print(f"  • Altitud: {info['current_position']['altitude_km']:.2f} km")
                        
                        print(f"\nElementos orbitales:")
                        oe = info['orbital_elements']
                        print(f"  • Inclinación: {oe['inclination_deg']:.2f}°")
                        print(f"  • Excentricidad: {oe['eccentricity']:.6f}")
                        print(f"  • Período orbital: {oe['period_hours']:.2f} horas")
                        print(f"  • Altitud aprox: {oe['approx_altitude_km']:.0f} km")
                        print(f"  • Revoluciones/día: {oe['mean_motion_rev_per_day']:.6f}")
                    else:
                        print("❌ Satélite no encontrado")
                        
            elif option == '4':
                # Calcular órbitas futuras
                sat_name = input("🚀 Nombre del satélite: ").strip()
                if sat_name:
                    try:
                        days = int(input("📅 Días hacia el futuro (máx 180): ") or "30")
                        days = min(days, 180)
                        
                        print(f"⏳ Calculando posiciones futuras para {days} días...")
                        positions = analyzer.calculate_future_positions(sat_name, days)
                        
                        if positions:
                            print(f"\n✅ Calculadas {len(positions)} posiciones")
                            print("Primeras 5 posiciones:")
                            for i, pos in enumerate(positions[:5]):
                                print(f"  {i+1}. {pos['datetime'].strftime('%Y-%m-%d %H:%M')} UTC")
                                print(f"     Lat: {pos['latitude']:.3f}°, Lon: {pos['longitude']:.3f}°")
                                print(f"     Alt: {pos['altitude_km']:.1f} km")
                        else:
                            print("❌ No se pudieron calcular las posiciones")
                    except ValueError:
                        print("❌ Número de días inválido")
                        
            elif option == '5':
                # Análisis de riesgo de colisión
                sat_name = input("⚠️  Nombre del satélite: ").strip()
                if sat_name:
                    try:
                        days = int(input("📅 Días a analizar (máx 180): ") or "180")
                        days = min(days, 180)
                        threshold = float(input("🎯 Distancia umbral en km (default 10): ") or "10")
                        
                        print(f"⏳ Analizando riesgo de colisión...")
                        risk_analysis = analyzer.analyze_collision_risk(sat_name, None, threshold, days)
                        
                        if 'error' not in risk_analysis:
                            print(f"\n🎯 ANÁLISIS DE RIESGO DE COLISIÓN")
                            print("-" * 50)
                            print(f"Satélite: {risk_analysis['satellite']}")
                            print(f"Período analizado: {risk_analysis['analysis_period_days']} días")
                            print(f"Satélites analizados: {risk_analysis['satellites_analyzed']}")
                            print(f"Umbral de distancia: {risk_analysis['threshold_km']} km")
                            print(f"NIVEL DE RIESGO: {risk_analysis['risk_level']}")
                            print(f"Encuentros cercanos: {risk_analysis['total_encounters']}")
                            
                            if risk_analysis['close_encounters']:
                                print("\n⚠️  ENCUENTROS CERCANOS DETECTADOS:")
                                for enc in risk_analysis['close_encounters'][:10]:  # Primeros 10
                                    print(f"  • {enc['datetime'].strftime('%Y-%m-%d %H:%M')} UTC")
                                    print(f"    Con: {enc['satellite2']}")
                                    print(f"    Distancia: {enc['distance_km']:.2f} km")
                            else:
                                print("✅ No se detectaron encuentros cercanos")
                        else:
                            print(f"❌ {risk_analysis['error']}")
                    except ValueError:
                        print("❌ Valores inválidos")
                        
            elif option == '6':
                # Visualizar órbita 2D
                sat_name = input("📈 Nombre del satélite: ").strip()
                if sat_name:
                    try:
                        hours = int(input("⏰ Horas de órbita a mostrar (default 24): ") or "24")
                        print("⏳ Generando visualización 2D...")
                        analyzer.plot_orbit(sat_name, hours)
                    except ValueError:
                        print("❌ Número de horas inválido")
                        
            elif option == '7':
                # Visualización 3D de la Tierra con satélites
                print("🌍 Visualización 3D de satélites alrededor de la Tierra")
                satellites_input = input("�️  Nombres de satélites (separados por coma): ").strip()
                if satellites_input:
                    satellite_names = [name.strip() for name in satellites_input.split(',')]
                    try:
                        hours = int(input("⏰ Horas de trayectoria (default 12): ") or "12")
                        print("⏳ Generando visualización 3D...")
                        analyzer.plot_3d_earth_with_satellites(satellite_names, hours)
                    except ValueError:
                        print("❌ Número de horas inválido")
                        
            elif option == '8':
                # Animación orbital 3D
                sat_name = input("🎬 Nombre del satélite para animar: ").strip()
                if sat_name:
                    try:
                        hours = int(input("⏰ Horas de órbita a animar (default 6): ") or "6")
                        frames = int(input("🎞️  Número de frames (default 50): ") or "50")
                        print("⏳ Generando animación 3D...")
                        analyzer.plot_orbital_animation(sat_name, hours, frames)
                    except ValueError:
                        print("❌ Valores inválidos")
                        
            elif option == '8':
                print("�👋 ¡Gracias por usar el Sistema de Análisis de Satélites!")
                break
                
            else:
                print("❌ Opción inválida. Selecciona 1-10.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")


if __name__ == "__main__":
    main()
