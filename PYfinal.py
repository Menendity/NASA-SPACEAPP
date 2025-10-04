#!/usr/bin/env python3
"""
Sistema de Análisis de Satélites
NASA Space App Challenge 2025

Este sistema permite:
1. Obtener datos de satélites desde Celestrak usando Skyfield
2. Buscar satélites por nombre
3. Calcular órbitas y posiciones futuras
4. Predecir posibles colisiones en los próximos 4 días
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
        try:
            if satellite_name not in self.satellites:
                print(f"❌ Satélite '{satellite_name}' no encontrado en la base de datos")
                # Buscar coincidencias parciales
                matches = [name for name in self.satellites.keys() if satellite_name.lower() in name.lower()]
                if matches:
                    print(f"💡 ¿Te refieres a alguno de estos?")
                    for i, match in enumerate(matches[:5], 1):
                        print(f"   {i}. {match}")
                return []
                
            satellite = self.satellites[satellite_name]['satellite']
            print(f"✅ Calculando posiciones para: {satellite_name}")
            
            # Crear timestamps para los próximos días
            start_time = self.ts.now()
            positions = []
            
            # Calcular posiciones cada 12 horas
            total_points = days_ahead * 2  # Cada 12 horas = 2 puntos por día
            print(f"📊 Calculando {total_points} posiciones para {days_ahead} días...")
            
            for hours in range(0, days_ahead * 24, 12):
                try:
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
                except Exception as calc_error:
                    print(f"⚠️  Error calculando posición para hora {hours}: {calc_error}")
                    continue
                    
            print(f"✅ Calculadas {len(positions)} posiciones exitosamente")
            return positions
            
        except Exception as e:
            print(f"❌ Error en calculate_future_positions: {str(e)}")
            return []
    
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
    
    def calculate_maneuver_time(self, v_rel: float, R_req: float = 1000.0, 
                              sigma_0: float = 100.0, k: float = 0.001, n: float = 3.0) -> Dict:
        """
        Calcular el tiempo necesario para iniciar maniobras de evasión de colisión
        
        Basado en la ecuación: t ≥ (R_req + n·σ₀) / (v_rel − n·k)
        
        Args:
            v_rel: Velocidad relativa entre objetos (m/s)
                  En LEO: ~100 m/s hasta ~14,000 m/s (encuentros frontales)
            R_req: Distancia de seguridad deseada (m). Ej: 100-1000 m
            sigma_0: Incertidumbre posicional actual (1-sigma, m)
            k: Tasa de crecimiento de incertidumbre (m/s)
            n: Factor de confianza (ej: 3 para 3σ)
            
        Returns:
            Dict: Análisis del tiempo de maniobra
        """
        try:
            # Validar parámetros de entrada
            if v_rel <= 0:
                return {'error': 'La velocidad relativa debe ser positiva'}
            
            if R_req <= 0:
                return {'error': 'La distancia de seguridad debe ser positiva'}
            
            if sigma_0 < 0:
                return {'error': 'La incertidumbre posicional no puede ser negativa'}
            
            # Calcular componentes de la ecuación
            numerador = R_req + n * sigma_0
            denominador = v_rel - n * k
            
            # Verificar que el denominador sea positivo
            if denominador <= 0:
                return {
                    'error': 'Configuración inválida',
                    'reason': 'La velocidad relativa es insuficiente comparada con el crecimiento de incertidumbre',
                    'recommendation': 'Reducir el factor de confianza (n) o mejorar la precisión orbital (reducir k)',
                    'v_rel': v_rel,
                    'n_k': n * k,
                    'deficit': abs(denominador)
                }
            
            # Calcular tiempo de maniobra
            t_maneuver_seconds = numerador / denominador
            
            # Convertir a diferentes unidades
            t_minutes = t_maneuver_seconds / 60
            t_hours = t_minutes / 60
            t_days = t_hours / 24
            
            # Determinar criticidad basada en el tiempo disponible
            if t_hours < 1:
                criticidad = "🔴 CRÍTICO"
                recomendacion = "Maniobra inmediata requerida"
            elif t_hours < 6:
                criticidad = "🟠 ALTO"
                recomendacion = "Preparar maniobra en las próximas horas"
            elif t_hours < 24:
                criticidad = "🟡 MEDIO"
                recomendacion = "Planificar maniobra para hoy"
            elif t_days < 7:
                criticidad = "🟢 BAJO"
                recomendacion = "Maniobra puede planificarse con anticipación"
            else:
                criticidad = "🔵 MÍNIMO"
                recomendacion = "Tiempo suficiente para análisis detallado"
            
            # Calcular escenarios alternativos
            escenarios = []
            
            # Escenario conservador (n=2)
            if n != 2:
                t_conservador = (R_req + 2 * sigma_0) / (v_rel - 2 * k) if (v_rel - 2 * k) > 0 else None
                if t_conservador:
                    escenarios.append({
                        'nombre': 'Conservador (2σ)',
                        'tiempo_segundos': t_conservador,
                        'tiempo_horas': t_conservador / 3600
                    })
            
            # Escenario agresivo (n=1)
            if n != 1:
                t_agresivo = (R_req + 1 * sigma_0) / (v_rel - 1 * k) if (v_rel - 1 * k) > 0 else None
                if t_agresivo:
                    escenarios.append({
                        'nombre': 'Agresivo (1σ)',
                        'tiempo_segundos': t_agresivo,
                        'tiempo_horas': t_agresivo / 3600
                    })
            
            # Análisis de sensibilidad
            sensibilidad = {
                'impacto_v_rel': {
                    'descripcion': 'Efecto de ±10% en velocidad relativa',
                    'v_rel_high': v_rel * 1.1,
                    't_high': (numerador) / (v_rel * 1.1 - n * k) if (v_rel * 1.1 - n * k) > 0 else None,
                    'v_rel_low': v_rel * 0.9,
                    't_low': (numerador) / (v_rel * 0.9 - n * k) if (v_rel * 0.9 - n * k) > 0 else None
                },
                'impacto_incertidumbre': {
                    'descripcion': 'Efecto de ±50% en incertidumbre',
                    'sigma_high': sigma_0 * 1.5,
                    't_sigma_high': (R_req + n * sigma_0 * 1.5) / denominador,
                    'sigma_low': sigma_0 * 0.5,
                    't_sigma_low': (R_req + n * sigma_0 * 0.5) / denominador
                }
            }
            
            return {
                'parametros': {
                    'v_rel_ms': v_rel,
                    'R_req_m': R_req,
                    'sigma_0_m': sigma_0,
                    'k_ms': k,
                    'factor_confianza': n
                },
                'tiempo_maniobra': {
                    'segundos': t_maneuver_seconds,
                    'minutos': t_minutes,
                    'horas': t_hours,
                    'dias': t_days
                },
                'evaluacion': {
                    'criticidad': criticidad,
                    'recomendacion': recomendacion
                },
                'componentes_calculo': {
                    'numerador': numerador,
                    'denominador': denominador,
                    'margen_seguridad': denominador - n * k
                },
                'escenarios_alternativos': escenarios,
                'analisis_sensibilidad': sensibilidad,
                'interpretacion': {
                    'contexto_leo': self._get_leo_context(v_rel),
                    'recomendaciones_operacionales': self._get_operational_recommendations(t_hours, v_rel)
                }
            }
            
        except Exception as e:
            return {'error': f'Error en cálculo: {str(e)}'}
    
    def _get_leo_context(self, v_rel: float) -> Dict:
        """Proporcionar contexto específico para órbitas LEO"""
        if v_rel < 500:
            tipo_encuentro = "Co-orbital o encuentro suave"
            descripcion = "Satélites en órbitas similares con baja velocidad relativa"
        elif v_rel < 2000:
            tipo_encuentro = "Encuentro lateral"
            descripcion = "Cruce de órbitas con ángulo moderado"
        elif v_rel < 8000:
            tipo_encuentro = "Encuentro perpendicular"
            descripcion = "Órbitas con planos orbitales diferentes"
        else:
            tipo_encuentro = "Encuentro frontal"
            descripcion = "Órbitas con inclinaciones opuestas - máximo riesgo"
            
        return {
            'tipo_encuentro': tipo_encuentro,
            'descripcion': descripcion,
            'velocidad_relativa_ms': v_rel,
            'velocidad_relativa_kmh': v_rel * 3.6
        }
    
    def _get_operational_recommendations(self, t_hours: float, v_rel: float) -> List[str]:
        """Generar recomendaciones operacionales específicas"""
        recomendaciones = []
        
        if t_hours < 1:
            recomendaciones.extend([
                "🚨 Activar protocolo de emergencia",
                "📡 Contactar inmediatamente con el centro de control",
                "⚡ Ejecutar maniobra de emergencia pre-programada",
                "📊 Monitoreo continuo de telemetría"
            ])
        elif t_hours < 6:
            recomendaciones.extend([
                "📋 Preparar plan de maniobra detallado",
                "🔍 Refinar datos orbitales con mediciones adicionales",
                "👥 Notificar a otros operadores satelitales",
                "⚙️ Verificar sistemas de propulsión"
            ])
        elif t_hours < 24:
            recomendaciones.extend([
                "📈 Realizar análisis de conjunción detallado",
                "🛰️ Considerar maniobras coordinadas si aplica",
                "📡 Incrementar frecuencia de tracking",
                "💾 Documentar procedimientos para caso similar"
            ])
        else:
            recomendaciones.extend([
                "🔬 Análisis exhaustivo de múltiples escenarios",
                "🤝 Coordinación con agencias espaciales",
                "📊 Optimización de combustible para maniobra",
                "🎯 Planificación de maniobra de precisión"
            ])
            
        # Recomendaciones específicas por velocidad relativa
        if v_rel > 10000:
            recomendaciones.append("⚠️ Encuentro de alta velocidad - considerar maniobra temprana")
        elif v_rel < 500:
            recomendaciones.append("🔄 Encuentro lento - maniobra de larga duración posible")
            
        return recomendaciones
    
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
    
    def comprehensive_collision_analysis(self, satellite1_name: str, satellite2_name: str = None,
                                       threshold_km: float = 10.0, days_ahead: int = 7) -> Dict:
        """
        Análisis completo de colisión incluyendo cálculo de tiempo de maniobra
        
        Args:
            satellite1_name: Primer satélite a analizar
            satellite2_name: Segundo satélite (si None, analiza contra muestra)
            threshold_km: Distancia mínima para considerar riesgo (km)
            days_ahead: Días a analizar hacia el futuro
            
        Returns:
            Dict: Análisis completo de colisión y tiempo de maniobra
        """
        print(f"🔍 Iniciando análisis completo de colisión para {satellite1_name}...")
        
        # Realizar análisis de colisión básico
        collision_analysis = self.analyze_collision_risk(
            satellite1_name, satellite2_name, threshold_km, days_ahead
        )
        
        if 'error' in collision_analysis:
            return collision_analysis
        
        # Si hay encuentros cercanos, calcular parámetros de maniobra
        maneuver_analyses = []
        
        if collision_analysis['close_encounters']:
            print(f"⚠️  {len(collision_analysis['close_encounters'])} encuentros cercanos detectados")
            
            for encounter in collision_analysis['close_encounters'][:5]:  # Analizar los primeros 5
                # Calcular velocidad relativa estimada para el encuentro
                sat1_pos = np.array(encounter['satellite1_pos'])
                sat2_pos = np.array(encounter['satellite2_pos'])
                distance_km = encounter['distance_km']
                
                # Estimar velocidad relativa basada en la órbita LEO típica
                # Para satélites LEO: velocidad orbital ~7.8 km/s
                orbital_velocity = 7800  # m/s
                
                # Estimar velocidad relativa basada en el tipo de encuentro
                if distance_km < 1:
                    # Encuentro muy cercano, probablemente frontal
                    v_rel_estimate = orbital_velocity * 1.8  # ~14,000 m/s
                elif distance_km < 5:
                    # Encuentro cercano, ángulo moderado  
                    v_rel_estimate = orbital_velocity * 1.2  # ~9,400 m/s
                else:
                    # Encuentro lejano, paralelo
                    v_rel_estimate = orbital_velocity * 0.2  # ~1,560 m/s
                
                # Parámetros típicos para análisis
                params_scenarios = [
                    {
                        'nombre': 'Conservador',
                        'R_req': 1000,  # 1 km de seguridad
                        'sigma_0': 200,  # 200 m de incertidumbre
                        'k': 0.002,     # Crecimiento moderado
                        'n': 3          # 3 sigma
                    },
                    {
                        'nombre': 'Estándar',
                        'R_req': 500,   # 500 m de seguridad
                        'sigma_0': 100,  # 100 m de incertidumbre
                        'k': 0.001,     # Crecimiento normal
                        'n': 2.5        # 2.5 sigma
                    },
                    {
                        'nombre': 'Agresivo',
                        'R_req': 200,   # 200 m de seguridad
                        'sigma_0': 50,   # 50 m de incertidumbre
                        'k': 0.0005,    # Bajo crecimiento
                        'n': 2          # 2 sigma
                    }
                ]
                
                encounter_maneuvers = []
                
                for scenario in params_scenarios:
                    maneuver_calc = self.calculate_maneuver_time(
                        v_rel=v_rel_estimate,
                        R_req=scenario['R_req'],
                        sigma_0=scenario['sigma_0'],
                        k=scenario['k'],
                        n=scenario['n']
                    )
                    
                    if 'error' not in maneuver_calc:
                        encounter_maneuvers.append({
                            'escenario': scenario['nombre'],
                            'parametros': scenario,
                            'tiempo_maniobra': maneuver_calc['tiempo_maniobra'],
                            'criticidad': maneuver_calc['evaluacion']['criticidad'],
                            'recomendacion': maneuver_calc['evaluacion']['recomendacion']
                        })
                
                maneuver_analyses.append({
                    'encuentro': {
                        'fecha': encounter['datetime'].strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'satelite_2': encounter['satellite2'],
                        'distancia_km': distance_km,
                        'velocidad_relativa_estimada': v_rel_estimate
                    },
                    'analisis_maniobra': encounter_maneuvers
                })
        
        # Generar recomendaciones generales
        recomendaciones_generales = self._generate_general_recommendations(
            collision_analysis, maneuver_analyses
        )
        
        # Calcular tiempo hasta primer encuentro
        tiempo_primer_encuentro = None
        if collision_analysis['close_encounters']:
            primer_encuentro = min(collision_analysis['close_encounters'], 
                                 key=lambda x: x['datetime'])
            tiempo_primer_encuentro = {
                'fecha': primer_encuentro['datetime'],
                'horas_restantes': (primer_encuentro['datetime'] - datetime.now()).total_seconds() / 3600,
                'distancia_km': primer_encuentro['distance_km']
            }
        
        return {
            'analisis_colision': collision_analysis,
            'analisis_maniobras': maneuver_analyses,
            'tiempo_primer_encuentro': tiempo_primer_encuentro,
            'recomendaciones_generales': recomendaciones_generales,
            'resumen_ejecutivo': self._generate_executive_summary(
                collision_analysis, maneuver_analyses, tiempo_primer_encuentro
            )
        }
    
    def _generate_general_recommendations(self, collision_analysis: Dict, 
                                        maneuver_analyses: List[Dict]) -> List[str]:
        """Generar recomendaciones generales basadas en el análisis"""
        recommendations = []
        
        risk_level = collision_analysis.get('risk_level', 'BAJO')
        total_encounters = collision_analysis.get('total_encounters', 0)
        
        if risk_level == 'CRÍTICO':
            recommendations.extend([
                "🚨 ALERTA CRÍTICA: Implementar protocolo de emergencia inmediatamente",
                "📡 Establecer comunicación continua con centros de control",
                "⚡ Preparar maniobra de emergencia automática",
                "🎯 Considerar múltiples opciones de maniobra"
            ])
        elif risk_level == 'ALTO':
            recommendations.extend([
                "⚠️ RIESGO ALTO: Planificar maniobra en las próximas 24 horas",
                "📊 Refinar datos orbitales con tracking adicional",
                "🤝 Coordinar con otros operadores si es necesario",
                "📋 Preparar plan de contingencia"
            ])
        elif risk_level == 'MEDIO':
            recommendations.extend([
                "🟡 RIESGO MEDIO: Monitoreo incrementado requerido",
                "📈 Análisis detallado de conjunción",
                "🔍 Evaluación de opciones de maniobra",
                "📅 Planificación preventiva"
            ])
        
        if total_encounters > 3:
            recommendations.append(f"📊 Múltiples encuentros ({total_encounters}) - considerar cambio orbital mayor")
        
        if maneuver_analyses:
            min_time = min([
                min([m['tiempo_maniobra']['horas'] for m in analysis['analisis_maniobra']])
                for analysis in maneuver_analyses if analysis['analisis_maniobra']
            ], default=float('inf'))
            
            if min_time < 1:
                recommendations.append("⏰ Tiempo de maniobra < 1 hora - Acción inmediata requerida")
            elif min_time < 6:
                recommendations.append("⏰ Tiempo de maniobra < 6 horas - Preparación urgente")
        
        return recommendations
    
    def _generate_executive_summary(self, collision_analysis: Dict, 
                                  maneuver_analyses: List[Dict], 
                                  primer_encuentro: Dict) -> Dict:
        """Generar resumen ejecutivo del análisis"""
        
        summary = {
            'satelite': collision_analysis.get('satellite', 'Desconocido'),
            'nivel_riesgo': collision_analysis.get('risk_level', 'BAJO'),
            'total_encuentros': collision_analysis.get('total_encounters', 0),
            'periodo_analisis_dias': collision_analysis.get('analysis_period_days', 0)
        }
        
        if primer_encuentro:
            summary['primer_encuentro'] = {
                'tiempo_horas': primer_encuentro['horas_restantes'],
                'distancia_km': primer_encuentro['distancia_km'],
                'fecha': primer_encuentro['fecha'].strftime('%Y-%m-%d %H:%M UTC')
            }
        
        if maneuver_analyses:
            # Tiempo mínimo de maniobra entre todos los escenarios
            tiempos_maniobra = []
            for analysis in maneuver_analyses:
                for maneuver in analysis['analisis_maniobra']:
                    tiempos_maniobra.append(maneuver['tiempo_maniobra']['horas'])
            
            if tiempos_maniobra:
                summary['tiempo_maniobra'] = {
                    'minimo_horas': min(tiempos_maniobra),
                    'maximo_horas': max(tiempos_maniobra),
                    'promedio_horas': sum(tiempos_maniobra) / len(tiempos_maniobra)
                }
        
        # Determinar acción recomendada
        if summary['nivel_riesgo'] == 'CRÍTICO':
            summary['accion_recomendada'] = "MANIOBRA INMEDIATA"
        elif summary['nivel_riesgo'] == 'ALTO':
            summary['accion_recomendada'] = "PREPARAR MANIOBRA URGENTE"
        elif summary['nivel_riesgo'] == 'MEDIO':
            summary['accion_recomendada'] = "MONITOREO INCREMENTADO"
        else:
            summary['accion_recomendada'] = "SEGUIMIENTO RUTINARIO"
        
        return summary
    
    def find_collision_cases(self, threshold_km: float = 50.0, days_ahead: int = 7, 
                           max_satellites: int = 500) -> List[Dict]:
        """
        Buscar casos reales de colisión entre satélites
        Función específica para encontrar encuentros cercanos reales
        
        Args:
            threshold_km: Distancia máxima para considerar encuentro cercano
            days_ahead: Días a analizar
            max_satellites: Máximo número de satélites a analizar
            
        Returns:
            List[Dict]: Lista de casos de colisión encontrados
        """
        print(f"🔍 BÚSQUEDA EXHAUSTIVA DE CASOS DE COLISIÓN")
        print(f"📊 Analizando hasta {max_satellites} satélites...")
        print(f"📏 Umbral: {threshold_km} km | 📅 Período: {days_ahead} días")
        print("-" * 60)
        
        collision_cases = []
        satellites_list = list(self.satellites.keys())
        
        # Analizar una muestra más grande de satélites
        sample_size = min(max_satellites, len(satellites_list))
        sample_satellites = satellites_list[:sample_size]
        
        analyzed_pairs = set()  # Evitar analizar el mismo par dos veces
        
        for i, sat1_name in enumerate(sample_satellites):
            if i % 50 == 0:  # Mostrar progreso cada 50 satélites
                progress = (i / sample_size) * 100
                print(f"📈 Progreso: {progress:.1f}% ({i}/{sample_size}) - Casos encontrados: {len(collision_cases)}")
            
            try:
                sat1 = self.satellites[sat1_name]['satellite']
                
                # Analizar contra una submuestra de otros satélites
                for j, sat2_name in enumerate(sample_satellites[i+1:i+51], i+1):  # Siguientes 50
                    if j >= len(sample_satellites):
                        break
                        
                    pair = tuple(sorted([sat1_name, sat2_name]))
                    if pair in analyzed_pairs:
                        continue
                    analyzed_pairs.add(pair)
                    
                    try:
                        sat2 = self.satellites[sat2_name]['satellite']
                        
                        # Verificar encuentros cada 2 horas para mayor precisión
                        for hours in range(0, days_ahead * 24, 2):
                            t = self.ts.tt_jd(self.ts.now().tt + hours / 24)
                            
                            pos1 = sat1.at(t)
                            pos2 = sat2.at(t)
                            
                            # Calcular distancia
                            distance_km = np.linalg.norm(
                                np.array(pos1.position.km) - np.array(pos2.position.km)
                            )
                            
                            if distance_km < threshold_km:
                                # ¡Encontramos un caso de colisión!
                                collision_cases.append({
                                    'satellite1': sat1_name,
                                    'satellite2': sat2_name,
                                    'datetime': t.utc_datetime(),
                                    'distance_km': distance_km,
                                    'hours_from_now': hours,
                                    'satellite1_pos': pos1.position.km,
                                    'satellite2_pos': pos2.position.km,
                                    'relative_velocity_estimated': self._estimate_relative_velocity(
                                        pos1.position.km, pos2.position.km, distance_km
                                    )
                                })
                                
                                print(f"🚨 CASO ENCONTRADO: {sat1_name} vs {sat2_name}")
                                print(f"   📅 {t.utc_datetime().strftime('%Y-%m-%d %H:%M')} UTC")
                                print(f"   📏 Distancia: {distance_km:.2f} km")
                                
                                # Si encontramos varios casos, no necesitamos más
                                if len(collision_cases) >= 5:
                                    print(f"✅ Suficientes casos encontrados. Deteniendo búsqueda.")
                                    return collision_cases
                                    
                    except Exception as e:
                        continue  # Continuar con el siguiente satélite
                        
            except Exception as e:
                continue  # Continuar con el siguiente satélite principal
        
        print(f"✅ Búsqueda completada. Casos encontrados: {len(collision_cases)}")
        return collision_cases
    
    def _estimate_relative_velocity(self, pos1: np.ndarray, pos2: np.ndarray, 
                                  distance_km: float) -> float:
        """Estimar velocidad relativa basada en posiciones y distancia"""
        # Velocidad orbital típica en LEO
        orbital_speed = 7800  # m/s
        
        # Estimar basado en la distancia del encuentro
        if distance_km < 5:
            return orbital_speed * 1.8  # Encuentro frontal probable
        elif distance_km < 20:
            return orbital_speed * 1.2  # Encuentro angular
        else:
            return orbital_speed * 0.5  # Encuentro lateral
    
    def demonstrate_collision_case(self) -> None:
        """
        Demostrar un caso de colisión encontrado con análisis completo
        """
        print("🔍 DEMOSTRACIÓN DE CASO DE COLISIÓN REAL")
        print("=" * 60)
        
        # Buscar casos de colisión
        cases = self.find_collision_cases(threshold_km=100, days_ahead=3, max_satellites=200)
        
        if not cases:
            print("❌ No se encontraron casos de colisión en la muestra analizada")
            print("💡 Esto puede ocurrir porque:")
            print("   • Los satélites están bien separados")
            print("   • La muestra analizada es pequeña")
            print("   • Los umbrales son muy estrictos")
            print("\n🎭 Generando caso simulado para demostración...")
            
            # Crear un caso simulado basado en datos reales
            self._create_simulated_case()
            return
        
        # Analizar el primer caso encontrado
        case = cases[0]
        print(f"\n🚨 CASO DE COLISIÓN DETECTADO:")
        print(f"🛰️  Satélite 1: {case['satellite1']}")
        print(f"🛰️  Satélite 2: {case['satellite2']}")
        print(f"📅 Fecha/Hora: {case['datetime'].strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"📏 Distancia: {case['distance_km']:.2f} km")
        print(f"⏰ En: {case['hours_from_now']} horas")
        
        # Calcular tiempo de maniobra para este caso
        v_rel = case['relative_velocity_estimated']
        print(f"\n⚡ ANÁLISIS DE TIEMPO DE MANIOBRA:")
        print(f"🚀 Velocidad relativa estimada: {v_rel:.0f} m/s")
        
        # Varios escenarios de maniobra
        scenarios = [
            {'name': 'Conservador', 'R_req': 2000, 'sigma_0': 200, 'k': 0.002, 'n': 3},
            {'name': 'Estándar', 'R_req': 1000, 'sigma_0': 100, 'k': 0.001, 'n': 2.5},
            {'name': 'Agresivo', 'R_req': 500, 'sigma_0': 50, 'k': 0.0008, 'n': 2}
        ]
        
        print(f"\n📊 ESCENARIOS DE MANIOBRA:")
        for scenario in scenarios:
            result = self.calculate_maneuver_time(
                v_rel=v_rel,
                R_req=scenario['R_req'],
                sigma_0=scenario['sigma_0'],
                k=scenario['k'],
                n=scenario['n']
            )
            
            if 'error' not in result:
                tiempo = result['tiempo_maniobra']
                print(f"   • {scenario['name']}: {tiempo['horas']:.2f} horas")
                print(f"     {result['evaluacion']['criticidad']}")
            else:
                print(f"   • {scenario['name']}: {result['error']}")
        
        # Mostrar todos los casos encontrados
        if len(cases) > 1:
            print(f"\n📋 OTROS CASOS DETECTADOS:")
            for i, other_case in enumerate(cases[1:], 2):
                print(f"   {i}. {other_case['satellite1']} vs {other_case['satellite2']}")
                print(f"      📅 {other_case['datetime'].strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"      📏 {other_case['distance_km']:.2f} km")
    
    def _create_simulated_case(self) -> None:
        """Crear un caso simulado basado en satélites reales"""
        print("🎭 CASO SIMULADO DE DEMOSTRACIÓN:")
        print("=" * 50)
        
        # Usar satélites reales para crear escenario creíble
        satellite_names = list(self.satellites.keys())
        sat1 = satellite_names[10] if len(satellite_names) > 10 else satellite_names[0]
        sat2 = satellite_names[50] if len(satellite_names) > 50 else satellite_names[1]
        
        import datetime
        future_time = datetime.datetime.now() + datetime.timedelta(hours=28, minutes=45)
        
        print(f"🛰️  Satélite 1: {sat1}")
        print(f"🛰️  Satélite 2: {sat2}")
        print(f"📅 Encuentro proyectado: {future_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"📏 Distancia mínima estimada: 15.3 km")
        print(f"🚀 Velocidad relativa: 8,200 m/s")
        print(f"⏰ Tiempo hasta encuentro: 28.75 horas")
        
        print(f"\n⚡ ANÁLISIS DE TIEMPO DE MANIOBRA:")
        result = self.calculate_maneuver_time(
            v_rel=8200,
            R_req=1000,
            sigma_0=120,
            k=0.001,
            n=3
        )
        
        if 'error' not in result:
            tiempo = result['tiempo_maniobra']
            print(f"⏰ Tiempo de maniobra requerido: {tiempo['horas']:.2f} horas")
            print(f"{result['evaluacion']['criticidad']}")
            print(f"💡 {result['evaluacion']['recomendacion']}")
            
            print(f"\n📊 EVALUACIÓN:")
            tiempo_disponible = 28.75
            tiempo_requerido = tiempo['horas']
            
            if tiempo_disponible > tiempo_requerido:
                margen = tiempo_disponible - tiempo_requerido
                print(f"✅ MARGEN SEGURO: {margen:.1f} horas disponibles")
                print(f"🎯 Ejecutar maniobra antes de: {(future_time - datetime.timedelta(hours=tiempo_requerido)).strftime('%Y-%m-%d %H:%M')} UTC")
            else:
                deficit = tiempo_requerido - tiempo_disponible
                print(f"🚨 SITUACIÓN CRÍTICA: Déficit de {deficit:.1f} horas")
                print(f"⚡ Maniobra inmediata requerida")
        
        print(f"\n💡 Este es un ejemplo de cómo el sistema detectaría y analizaría")
        print(f"   un caso real de conjunción satelital.")


# NUEVO MÓDULO PARA EL HACKATÓN - SISTEMA ISL CONTROL
class ISLControlSystem:
    """
    Sistema de Control de Enlaces Inter-Satelitales (ISL) con conciencia de propulsión
    
    Este módulo simula la lógica que se ejecutaría en el chip IENAI para:
    - Gestionar el tráfico de red satelital basado en riesgo de colisión
    - Optimizar el enrutamiento considerando el estado del propulsor
    - Tomar decisiones autónomas de maniobra y comunicación
    """
    
    def __init__(self, analyzer: SatelliteAnalyzer):
        self.analyzer = analyzer
        self.network_nodes = []  # Lista de satélites en la red
        self.current_routes = {}  # Rutas actuales de comunicación
        
    def determine_thrust_aware_routing(self, sat_local_name: str, sat_neighbor_name: str, 
                                       collision_risk_data: Dict, propellant_level: float) -> Dict:
        """
        Simula la lógica de enrutamiento basada en el riesgo de colisión y el estado del propulsor IENAI.
        ESTA FUNCIÓN SE EJECUTARÍA EN EL CHIP DEL IENAI.
        
        Args:
            sat_local_name: Nombre del satélite local (este satélite)
            sat_neighbor_name: Satélite vecino en la red
            collision_risk_data: Datos de riesgo de colisión
            propellant_level: Nivel de propelente (0.0 a 1.0)
            
        Returns:
            Dict: Comandos y decisiones del sistema ISL
        """
        
        # 1. Evaluar si se necesita una maniobra (usando la lógica existente)
        risk_level = collision_risk_data.get('risk_level', 'BAJO')
        close_encounters = collision_risk_data.get('close_encounters', [])
        
        # 2. Calcular parámetros de maniobra basados en el riesgo
        maneuver_analysis = None
        time_to_maneuver_hours = float('inf')
        
        if risk_level in ['ALTO', 'CRÍTICO'] and close_encounters:
            # Obtener el encuentro más cercano
            nearest_encounter = min(close_encounters, key=lambda x: x['distance_km'])
            
            # Estimar velocidad relativa basada en la distancia del encuentro
            if nearest_encounter['distance_km'] < 5:
                v_rel_estimate = 12000  # Encuentro frontal crítico
            elif nearest_encounter['distance_km'] < 20:
                v_rel_estimate = 8000   # Encuentro perpendicular
            else:
                v_rel_estimate = 3000   # Encuentro lateral
            
            # Calcular tiempo de maniobra requerido
            maneuver_analysis = self.analyzer.calculate_maneuver_time(
                v_rel=v_rel_estimate,
                R_req=500.0,     # 500m de seguridad para satélites comerciales
                sigma_0=100.0,   # 100m de incertidumbre estándar
                k=0.001,         # Crecimiento normal de incertidumbre
                n=3.0            # 3 sigma de confianza
            )
            
            if 'error' not in maneuver_analysis:
                time_to_maneuver_hours = maneuver_analysis['tiempo_maniobra']['horas']
        
        # 3. LÓGICA DE DECISIÓN ISL (El corazón del proyecto)
        decision_result = self._make_isl_decision(
            sat_local_name, sat_neighbor_name, risk_level, 
            time_to_maneuver_hours, propellant_level, maneuver_analysis
        )
        
        return decision_result
    
    def _make_isl_decision(self, sat_local: str, sat_neighbor: str, risk_level: str,
                          time_hours: float, propellant: float, maneuver_data: Dict) -> Dict:
        """
        Núcleo de la lógica de decisión ISL
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clasificar urgencia temporal
        if time_hours < 1:
            urgency = "CRÍTICO_INMEDIATO"
        elif time_hours < 6:
            urgency = "CRÍTICO_CORTO_PLAZO"
        elif time_hours < 24:
            urgency = "MODERADO"
        else:
            urgency = "BAJO"
        
        # DECISION TREE PRINCIPAL
        if urgency in ["CRÍTICO_INMEDIATO", "CRÍTICO_CORTO_PLAZO"]:
            if propellant > 0.15:  # Suficiente combustible (>15%)
                command = "THRUST_IMMINENT"
                action = f"Preparando maniobra de evasión. Desviando tráfico crítico al satélite {sat_neighbor}"
                network_priority = "HIGH_REROUTE"
                bandwidth_allocation = 0.2  # 20% del ancho de banda para coordinar maniobra
                
            elif propellant > 0.05:  # Combustible limitado (5-15%)
                command = "THRUST_CONDITIONAL"
                action = f"Maniobra condicional. Evaluando alternativas. Alertando a {sat_neighbor}"
                network_priority = "MEDIUM_REROUTE"
                bandwidth_allocation = 0.1  # 10% para coordinación
                
            else:  # Combustible insuficiente (<5%)
                command = "THRUST_IMPOSSIBLE"
                action = f"Combustible insuficiente. Emitiendo alerta de posición. Transferencia total a {sat_neighbor}"
                network_priority = "EMERGENCY_REROUTE"
                bandwidth_allocation = 0.05  # 5% mínimo para alertas
                
        elif urgency == "MODERADO":
            if propellant > 0.25:  # Buen nivel de combustible
                command = "THRUST_PLANNED"
                action = f"Maniobra planificada. Coordinando con {sat_neighbor} para redistribución de tráfico"
                network_priority = "PLANNED_REROUTE"
                bandwidth_allocation = 0.8  # 80% operación normal
                
            else:
                command = "THRUST_PRESERVE"
                action = f"Conservando combustible. Solicitando soporte de red a {sat_neighbor}"
                network_priority = "FUEL_CONSERVATION"
                bandwidth_allocation = 0.6  # 60% operación reducida
                
        else:  # BAJO riesgo
            command = "ROUTE_NORMAL"
            action = "Operación normal. Sin amenaza inmediata de colisión"
            network_priority = "NORMAL"
            bandwidth_allocation = 1.0  # 100% operación normal
        
        # Generar protocolo de comunicación ISL
        isl_protocol = self._generate_isl_protocol(
            command, sat_local, sat_neighbor, urgency, propellant
        )
        
        return {
            'timestamp': timestamp,
            'command': command,
            'action': action,
            'urgency_level': urgency,
            'risk_assessment': risk_level,
            'propellant_status': f"{propellant*100:.1f}%",
            'time_to_maneuver_hours': time_hours,
            'network_priority': network_priority,
            'bandwidth_allocation': bandwidth_allocation,
            'target_satellite': sat_neighbor,
            'isl_protocol': isl_protocol,
            'maneuver_data': maneuver_data,
            'autonomous_decision': True,
            'chip_location': 'IENAI_PROCESSOR'
        }
    
    def _generate_isl_protocol(self, command: str, sat_local: str, sat_neighbor: str,
                              urgency: str, propellant: float) -> Dict:
        """
        Generar protocolo de comunicación entre satélites
        """
        protocol = {
            'message_type': 'ISL_COORDINATION',
            'source': sat_local,
            'destination': sat_neighbor,
            'priority': 'HIGH' if urgency.startswith('CRÍTICO') else 'MEDIUM',
            'encryption': 'AES256_QUANTUM_SAFE',
            'compression': 'SATELLITE_OPTIMIZED'
        }
        
        if command == "THRUST_IMMINENT":
            protocol['payload'] = {
                'alert_type': 'IMMINENT_MANEUVER',
                'maneuver_window': '< 1 hour',
                'requested_action': 'TAKE_TRAFFIC_LOAD',
                'backup_required': True,
                'telemetry_sharing': True
            }
        elif command == "THRUST_IMPOSSIBLE":
            protocol['payload'] = {
                'alert_type': 'PROPULSION_FAILURE',
                'maneuver_capability': False,
                'requested_action': 'EMERGENCY_BACKUP',
                'position_alert': True,
                'ground_notification': True
            }
        elif command == "ROUTE_NORMAL":
            protocol['payload'] = {
                'alert_type': 'STATUS_NORMAL',
                'maneuver_capability': True,
                'requested_action': 'MAINTAIN_NORMAL_OPS',
                'health_check': True
            }
        else:
            protocol['payload'] = {
                'alert_type': 'CONDITIONAL_MANEUVER',
                'maneuver_probability': f"{min(1.0, (1.0 - propellant) + 0.5):.2f}",
                'requested_action': 'STANDBY_SUPPORT',
                'monitoring_required': True
            }
        
        return protocol
    
    def simulate_constellation_response(self, decision_result: Dict, 
                                      constellation_size: int = 5) -> Dict:
        """
        Simular la respuesta de una constelación de satélites al comando ISL
        """
        constellation_response = {
            'constellation_id': 'IENAI_NETWORK_ALPHA',
            'total_satellites': constellation_size,
            'responding_satellites': [],
            'network_adaptation': {},
            'collective_decision': None
        }
        
        # Simular respuesta de otros satélites
        for i in range(constellation_size):
            sat_id = f"IENAI_SAT_{i+1:02d}"
            if sat_id != decision_result.get('target_satellite', ''):
                
                # Simular capacidad de cada satélite
                sat_capacity = np.random.uniform(0.6, 1.0)  # 60-100% capacidad
                sat_fuel = np.random.uniform(0.1, 0.9)      # 10-90% combustible
                
                response = {
                    'satellite_id': sat_id,
                    'available_capacity': f"{sat_capacity*100:.1f}%",
                    'fuel_level': f"{sat_fuel*100:.1f}%",
                    'can_assist': sat_capacity > 0.3,
                    'priority_level': 'HIGH' if sat_capacity > 0.7 else 'MEDIUM'
                }
                
                constellation_response['responding_satellites'].append(response)
        
        # Calcular adaptación de red
        total_capacity = sum([float(sat['available_capacity'].rstrip('%'))/100 
                            for sat in constellation_response['responding_satellites']])
        
        constellation_response['network_adaptation'] = {
            'total_available_capacity': f"{total_capacity*100:.1f}%",
            'load_distribution': 'AUTOMATIC',
            'failover_ready': total_capacity > 1.5,
            'latency_impact': 'MINIMAL' if total_capacity > 2.0 else 'MODERATE'
        }
        
        # Decisión colectiva de la constelación
        if decision_result['urgency_level'].startswith('CRÍTICO'):
            constellation_response['collective_decision'] = 'EMERGENCY_PROTOCOL_ACTIVATED'
        elif total_capacity > 1.8:
            constellation_response['collective_decision'] = 'FULL_SUPPORT_GRANTED'
        else:
            constellation_response['collective_decision'] = 'LIMITED_SUPPORT_AVAILABLE'
        
        return constellation_response


class HackathonDemo:
    """
    Clase para demostrar el sistema ISL en el hackathon
    """
    
    def __init__(self, analyzer: SatelliteAnalyzer):
        self.analyzer = analyzer
        self.isl_system = ISLControlSystem(analyzer)
        
    def run_complete_demo(self):
        """
        Ejecutar demostración completa del sistema ISL para el hackathon
        """
        print("🚀 DEMOSTRACIÓN COMPLETA DEL SISTEMA ISL-IENAI")
        print("=" * 70)
        print("🎯 Sistema de Control de Enlaces Inter-Satelitales con Conciencia de Propulsión")
        print("💡 Simulando operación autónoma en chip IENAI")
        print("-" * 70)
        
        # Escenarios de prueba
        scenarios = [
            {
                'name': '🔴 ESCENARIO CRÍTICO: Encuentro Frontal Inminente',
                'risk_data': {
                    'risk_level': 'CRÍTICO',
                    'close_encounters': [{'distance_km': 2.5, 'datetime': datetime.now()}]
                },
                'propellant': 0.85,  # 85% combustible
                'description': 'Satélite con buen combustible detecta colisión inminente'
            },
            {
                'name': '🟠 ESCENARIO CRÍTICO: Combustible Bajo',
                'risk_data': {
                    'risk_level': 'ALTO',
                    'close_encounters': [{'distance_km': 8.3, 'datetime': datetime.now()}]
                },
                'propellant': 0.03,  # 3% combustible
                'description': 'Satélite con combustible crítico detecta amenaza'
            },
            {
                'name': '🟡 ESCENARIO MODERADO: Encuentro Planificado',
                'risk_data': {
                    'risk_level': 'MEDIO',
                    'close_encounters': [{'distance_km': 25.7, 'datetime': datetime.now()}]
                },
                'propellant': 0.60,  # 60% combustible
                'description': 'Encuentro detectado con tiempo para planificar'
            },
            {
                'name': '🟢 ESCENARIO NORMAL: Operación Rutinaria',
                'risk_data': {
                    'risk_level': 'BAJO',
                    'close_encounters': []
                },
                'propellant': 0.75,  # 75% combustible
                'description': 'Operación normal sin amenazas detectadas'
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   📝 {scenario['description']}")
            print(f"   ⛽ Combustible: {scenario['propellant']*100:.1f}%")
            
            # Ejecutar análisis ISL
            decision = self.isl_system.determine_thrust_aware_routing(
                sat_local_name="IENAI_SAT_01",
                sat_neighbor_name="IENAI_SAT_02", 
                collision_risk_data=scenario['risk_data'],
                propellant_level=scenario['propellant']
            )
            
            # Mostrar resultados
            print(f"   🤖 DECISIÓN AUTÓNOMA: {decision['command']}")
            print(f"   ⚡ Acción: {decision['action']}")
            print(f"   📡 Prioridad de red: {decision['network_priority']}")
            print(f"   📊 Ancho de banda: {decision['bandwidth_allocation']*100:.0f}%")
            
            if decision['time_to_maneuver_hours'] < float('inf'):
                print(f"   ⏰ Tiempo para maniobra: {decision['time_to_maneuver_hours']:.2f} horas")
            
            # Simular respuesta de constelación
            constellation_response = self.isl_system.simulate_constellation_response(decision)
            print(f"   🛰️  Respuesta de constelación: {constellation_response['collective_decision']}")
            print(f"   🌐 Capacidad disponible: {constellation_response['network_adaptation']['total_available_capacity']}")
            
            print("   " + "-" * 50)
        
        print(f"\n✅ DEMOSTRACIÓN COMPLETADA")
        print(f"🎯 El sistema ISL-IENAI está listo para:")
        print(f"   • Detección autónoma de riesgos de colisión")
        print(f"   • Toma de decisiones basada en estado de propulsión")
        print(f"   • Gestión inteligente de red satelital")
        print(f"   • Coordinación de constelación en tiempo real")
        print(f"   • Operación completamente autónoma en el espacio")
    
    
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
    print("  10. Cálculo de tiempo de maniobra de evasión")
    print("  11. Análisis completo de colisión + maniobra")
    print("  12. 🔍 BUSCAR CASOS REALES DE COLISIÓN")
    print("  13. 🚀 DEMO SISTEMA ISL-IENAI (HACKATHON)")
    print("  14. 🤖 Simulador ISL Individual")
    print("  15. Salir")
    print("=" * 60)


def main():
    """Función principal del programa"""
    print("=" * 60)
    print("🛰️  SISTEMA DE ANÁLISIS DE SATÉLITES")
    print("    NASA Space App Challenge 2025 - Malkie Space")
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
            option = input("Selecciona una opción (1-15): ").strip()
            
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
                print("🌟 Satélites populares por categoría:")
                popular = analyzer.get_popular_satellites()
                for category, satellites in popular.items():
                    print(f"\n📂 {category.upper()}:")
                    for i, name in enumerate(satellites, 1):
                        print(f"   {i}. {name}")
                        
            elif option == '3':
                # Información detallada de un satélite
                sat_name = input("📋 Nombre del satélite: ").strip()
                if sat_name:
                    info = analyzer.get_satellite_info(sat_name)
                    if 'error' not in info:
                        print(f"\n🛰️  INFORMACIÓN DETALLADA: {sat_name}")
                        print("=" * 50)
                        print(f"📅 Fecha de los datos: {info['current_time']}")
                        print(f"📍 Posición actual:")
                        print(f"   • Latitud: {info['position']['latitude']:.3f}°")
                        print(f"   • Longitud: {info['position']['longitude']:.3f}°")
                        print(f"   • Altitud: {info['position']['altitude']:.1f} km")
                        print(f"📊 Parámetros orbitales:")
                        print(f"   • Inclinación: {info['orbital_elements'].get('inclination', 'N/A')}")
                        print(f"   • Excentricidad: {info['orbital_elements'].get('eccentricity', 'N/A')}")
                        print(f"   • Período: {info['orbital_elements'].get('period_minutes', 'N/A')} min")
                    else:
                        print(f"❌ {info['error']}")
                        
            elif option == '4':
                # Calcular órbitas futuras
                sat_name = input("🚀 Nombre del satélite: ").strip()
                if sat_name:
                    try:
                        days = int(input("📅 Días a calcular (default 7): ") or "7")
                        days = min(days, 180)  # Limitar a máximo 180 días
                        print(f"⏳ Calculando posiciones futuras para {days} días...")
                        positions = analyzer.calculate_future_positions(sat_name, days)
                        
                        if positions:
                            print(f"\n🛰️  PREDICCIONES ORBITALES: {sat_name}")
                            print("=" * 60)
                            for pos in positions[:20]:  # Mostrar primeros 20
                                print(f"📅 {pos['datetime'].strftime('%Y-%m-%d %H:%M')} UTC")
                                print(f"   Lat: {pos['latitude']:7.3f}°  Lon: {pos['longitude']:8.3f}°  Alt: {pos['altitude_km']:7.1f} km")
                            
                            if len(positions) > 20:
                                print(f"   ... y {len(positions) - 20} predicciones más")
                                
                            # Mostrar estadísticas
                            altitudes = [pos['altitude_km'] for pos in positions]
                            print(f"\n📈 ESTADÍSTICAS:")
                            print(f"   • Altitud mínima: {min(altitudes):.1f} km")
                            print(f"   • Altitud máxima: {max(altitudes):.1f} km")
                            print(f"   • Altitud promedio: {sum(altitudes)/len(altitudes):.1f} km")
                        else:
                            print("❌ No se pudieron calcular las posiciones")
                            print("💡 Sugerencias:")
                            print("   • Verifica que el nombre del satélite sea exacto")
                            print("   • Usa la opción 1 para buscar satélites disponibles")
                            print("   • Intenta con nombres populares como: ISS (ZARYA), STARLINK-1007")
                    except ValueError:
                        print("❌ Número de días inválido. Debe ser un número entero.")
                        
            elif option == '5':
                # Analizar riesgo de colisión
                sat_name = input("⚠️  Satélite principal: ").strip()
                if sat_name:
                    sat2_name = input("🎯 Segundo satélite (Enter para analizar contra todos): ").strip() or None
                    try:
                        threshold = float(input("📏 Distancia mínima en km (default 10): ") or "10")
                        days = int(input("📅 Días a analizar (4): ") or "4")
                        
                        print("⏳ Analizando riesgo de colisión...")
                        risk_analysis = analyzer.analyze_collision_risk(sat_name, sat2_name, threshold, days)
                        
                        if 'error' not in risk_analysis:
                            print(f"\n⚠️  ANÁLISIS DE RIESGO DE COLISIÓN")
                            print("=" * 50)
                            print(f"🛰️  Satélite: {risk_analysis['satellite']}")
                            print(f"📊 Nivel de riesgo: {risk_analysis['risk_level']}")
                            print(f"📈 Encuentros cercanos: {risk_analysis['total_encounters']}")
                            print(f"📅 Período analizado: {risk_analysis['analysis_period_days']} días")
                            print(f"📏 Umbral: {risk_analysis['threshold_km']} km")
                            
                            if risk_analysis['close_encounters']:
                                print(f"\n🚨 ENCUENTROS CERCANOS DETECTADOS:")
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
                satellites_input = input("🛰️  Nombres de satélites (separados por coma): ").strip()
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
                        
            elif option == '9':
                # Exportar lista completa de satélites
                filename = input("📁 Nombre del archivo (default: satelites_disponibles.txt): ").strip() or "satelites_disponibles.txt"
                print("⏳ Exportando lista de satélites...")
                if analyzer.export_satellites_list(filename):
                    print(f"✅ Lista exportada exitosamente a: {filename}")
                else:
                    print("❌ Error al exportar la lista")
                    
            elif option == '10':
                # Cálculo de tiempo de maniobra de evasión
                print("⏰ CÁLCULO DE TIEMPO DE MANIOBRA DE EVASIÓN")
                print("=" * 50)
                try:
                    v_rel = float(input("🚀 Velocidad relativa (m/s) [100-14000]: "))
                    R_req = float(input("📏 Distancia de seguridad (m) [default 1000]: ") or "1000")
                    sigma_0 = float(input("📊 Incertidumbre posicional (m) [default 100]: ") or "100")
                    k = float(input("📈 Tasa crecimiento incertidumbre (m/s) [default 0.001]: ") or "0.001")
                    n = float(input("🎯 Factor de confianza (sigma) [default 3]: ") or "3")
                    
                    result = analyzer.calculate_maneuver_time(v_rel, R_req, sigma_0, k, n)
                    
                    if 'error' not in result:
                        print(f"\n⏰ RESULTADO DEL ANÁLISIS DE MANIOBRA")
                        print("=" * 50)
                        print(f"⚡ Tiempo de maniobra requerido:")
                        print(f"   • {result['tiempo_maniobra']['segundos']:.1f} segundos")
                        print(f"   • {result['tiempo_maniobra']['minutos']:.1f} minutos")
                        print(f"   • {result['tiempo_maniobra']['horas']:.2f} horas")
                        print(f"   • {result['tiempo_maniobra']['dias']:.3f} días")
                        
                        print(f"\n{result['evaluacion']['criticidad']}")
                        print(f"💡 {result['evaluacion']['recomendacion']}")
                        
                        print(f"\n🎯 Contexto del encuentro:")
                        print(f"   • {result['interpretacion']['contexto_leo']['tipo_encuentro']}")
                        print(f"   • {result['interpretacion']['contexto_leo']['descripcion']}")
                        
                        print(f"\n📋 Recomendaciones operacionales:")
                        for rec in result['interpretacion']['recomendaciones_operacionales']:
                            print(f"   {rec}")
                            
                        if result['escenarios_alternativos']:
                            print(f"\n📊 Escenarios alternativos:")
                            for escenario in result['escenarios_alternativos']:
                                print(f"   • {escenario['nombre']}: {escenario['tiempo_horas']:.2f} horas")
                    else:
                        print(f"❌ {result['error']}")
                        if 'recommendation' in result:
                            print(f"💡 {result['recommendation']}")
                            
                except ValueError:
                    print("❌ Valores inválidos. Asegúrate de ingresar números válidos.")
                    
            elif option == '11':
                # Análisis completo de colisión + maniobra
                print("🔍 ANÁLISIS COMPLETO: COLISIÓN + MANIOBRA")
                print("=" * 50)
                sat_name = input("🛰️  Nombre del satélite principal: ").strip()
                if sat_name:
                    sat2_name = input("🎯 Segundo satélite (Enter para analizar muestra): ").strip() or None
                    try:
                        threshold = float(input("📏 Distancia mínima en km (default 10): ") or "10")
                        days = int(input("📅 Días a analizar (default 7): ") or "7")
                        
                        print("⏳ Realizando análisis completo...")
                        comprehensive = analyzer.comprehensive_collision_analysis(
                            sat_name, sat2_name, threshold, days
                        )
                        
                        if 'error' not in comprehensive:
                            # Mostrar resumen ejecutivo
                            summary = comprehensive['resumen_ejecutivo']
                            print(f"\n📊 RESUMEN EJECUTIVO")
                            print("=" * 40)
                            print(f"🛰️  Satélite: {summary['satelite']}")
                            print(f"⚠️  Nivel de riesgo: {summary['nivel_riesgo']}")
                            print(f"📈 Total encuentros: {summary['total_encuentros']}")
                            print(f"🎯 Acción recomendada: {summary['accion_recomendada']}")
                            
                            if summary.get('primer_encuentro'):
                                pe = summary['primer_encuentro']
                                print(f"\n⏰ PRIMER ENCUENTRO:")
                                print(f"   • Fecha: {pe['fecha']}")
                                print(f"   • En: {pe['tiempo_horas']:.1f} horas")
                                print(f"   • Distancia: {pe['distancia_km']:.2f} km")
                            
                            if summary.get('tiempo_maniobra'):
                                tm = summary['tiempo_maniobra']
                                print(f"\n⚡ TIEMPO DE MANIOBRA:")
                                print(f"   • Mínimo: {tm['minimo_horas']:.2f} horas")
                                print(f"   • Máximo: {tm['maximo_horas']:.2f} horas")
                                print(f"   • Promedio: {tm['promedio_horas']:.2f} horas")
                            
                            # Mostrar recomendaciones generales
                            if comprehensive['recomendaciones_generales']:
                                print(f"\n💡 RECOMENDACIONES GENERALES:")
                                for rec in comprehensive['recomendaciones_generales']:
                                    print(f"   {rec}")
                            
                            # Mostrar análisis detallado de maniobras si hay encuentros
                            if comprehensive['analisis_maniobras']:
                                print(f"\n📊 ANÁLISIS DETALLADO DE MANIOBRAS:")
                                for i, analysis in enumerate(comprehensive['analisis_maniobras'][:3], 1):
                                    encounter = analysis['encuentro']
                                    print(f"\n   {i}. Encuentro: {encounter['fecha']}")
                                    print(f"      Con: {encounter['satelite_2']}")
                                    print(f"      Distancia: {encounter['distancia_km']:.2f} km")
                                    print(f"      V_rel estimada: {encounter['velocidad_relativa_estimada']:.0f} m/s")
                                    
                                    for maneuver in analysis['analisis_maniobra']:
                                        print(f"      • {maneuver['escenario']}: {maneuver['tiempo_maniobra']['horas']:.2f} horas")
                                        print(f"        {maneuver['criticidad']}")
                        else:
                            print(f"❌ {comprehensive['error']}")
                            
                    except ValueError:
                        print("❌ Valores inválidos")
                        
            elif option == '12':
                # Buscar casos reales de colisión
                print("🔍 BÚSQUEDA EXHAUSTIVA DE CASOS DE COLISIÓN")
                print("=" * 50)
                print("💡 Esta función buscará casos reales de encuentros cercanos")
                print("   entre satélites en la base de datos actual.")
                print()
                
                try:
                    threshold = float(input("📏 Umbral de distancia en km (default 75): ") or "75")
                    days = int(input("📅 Días a analizar (default 3): ") or "3")
                    max_sats = int(input("🛰️  Máximo satélites a analizar (default 300): ") or "300")
                    
                    print("\n⏳ Iniciando búsqueda exhaustiva...")
                    print("⚠️  Esta operación puede tomar varios minutos...")
                    
                    # Ejecutar búsqueda de casos de colisión
                    analyzer.demonstrate_collision_case()
                    
                except ValueError:
                    print("❌ Valores inválidos")
                except KeyboardInterrupt:
                    print("\n⏹️  Búsqueda cancelada por el usuario")
                    
            elif option == '13':
                # Demo completo del sistema ISL-IENAI para hackathon
                print("🚀 INICIANDO DEMOSTRACIÓN DEL SISTEMA ISL-IENAI")
                print("=" * 60)
                print("💡 Sistema de Control de Enlaces Inter-Satelitales")
                print("🎯 Demostrando toma de decisiones autónomas en el espacio")
                print()
                
                try:
                    demo = HackathonDemo(analyzer)
                    demo.run_complete_demo()
                except Exception as e:
                    print(f"❌ Error en demostración: {str(e)}")
                    
            elif option == '14':
                # Simulador ISL individual
                print("🤖 SIMULADOR ISL INDIVIDUAL")
                print("=" * 50)
                print("💡 Configura tu propio escenario de análisis ISL")
                print()
                
                try:
                    sat_local = input("🛰️  Satélite local (default: IENAI_SAT_01): ").strip() or "IENAI_SAT_01"
                    sat_neighbor = input("📡 Satélite vecino (default: IENAI_SAT_02): ").strip() or "IENAI_SAT_02"
                    
                    print("\n🎯 Configurar escenario de riesgo:")
                    print("   1. Riesgo CRÍTICO (encuentro < 5 km)")
                    print("   2. Riesgo ALTO (encuentro 5-20 km)")
                    print("   3. Riesgo MEDIO (encuentro 20-50 km)")
                    print("   4. Riesgo BAJO (sin amenazas)")
                    
                    risk_choice = input("Selecciona nivel de riesgo (1-4): ").strip()
                    propellant = float(input("⛽ Nivel de combustible (0.0-1.0): ") or "0.5")
                    
                    # Configurar datos de riesgo según la selección
                    risk_configs = {
                        '1': {'risk_level': 'CRÍTICO', 'close_encounters': [{'distance_km': 2.1, 'datetime': datetime.now()}]},
                        '2': {'risk_level': 'ALTO', 'close_encounters': [{'distance_km': 12.5, 'datetime': datetime.now()}]},
                        '3': {'risk_level': 'MEDIO', 'close_encounters': [{'distance_km': 35.0, 'datetime': datetime.now()}]},
                        '4': {'risk_level': 'BAJO', 'close_encounters': []}
                    }
                    
                    risk_data = risk_configs.get(risk_choice, risk_configs['4'])
                    
                    # Ejecutar análisis ISL
                    isl_system = ISLControlSystem(analyzer)
                    result = isl_system.determine_thrust_aware_routing(
                        sat_local, sat_neighbor, risk_data, propellant
                    )
                    
                    # Mostrar resultados detallados
                    print(f"\n🤖 RESULTADO DEL ANÁLISIS ISL:")
                    print("=" * 50)
                    print(f"⏰ Timestamp: {result['timestamp']}")
                    print(f"🚀 Comando: {result['command']}")
                    print(f"⚡ Acción: {result['action']}")
                    print(f"🎯 Urgencia: {result['urgency_level']}")
                    print(f"📊 Riesgo: {result['risk_assessment']}")
                    print(f"⛽ Combustible: {result['propellant_status']}")
                    
                    if result['time_to_maneuver_hours'] < float('inf'):
                        print(f"⏰ Tiempo para maniobra: {result['time_to_maneuver_hours']:.3f} horas")
                    
                    print(f"📡 Prioridad de red: {result['network_priority']}")
                    print(f"📶 Ancho de banda: {result['bandwidth_allocation']*100:.0f}%")
                    print(f"🎯 Satélite objetivo: {result['target_satellite']}")
                    print(f"🧠 Decisión autónoma: {result['autonomous_decision']}")
                    print(f"💻 Ubicación: {result['chip_location']}")
                    
                    # Mostrar protocolo ISL
                    protocol = result['isl_protocol']
                    print(f"\n📡 PROTOCOLO ISL:")
                    print(f"   • Tipo: {protocol['message_type']}")
                    print(f"   • Prioridad: {protocol['priority']}")
                    print(f"   • Encriptación: {protocol['encryption']}")
                    print(f"   • Acción solicitada: {protocol['payload']['requested_action']}")
                    
                    # Simular respuesta de constelación
                    constellation = isl_system.simulate_constellation_response(result)
                    print(f"\n🌐 RESPUESTA DE CONSTELACIÓN:")
                    print(f"   • Decisión colectiva: {constellation['collective_decision']}")
                    print(f"   • Capacidad total: {constellation['network_adaptation']['total_available_capacity']}")
                    print(f"   • Satélites respondiendo: {len(constellation['responding_satellites'])}")
                    print(f"   • Failover listo: {constellation['network_adaptation']['failover_ready']}")
                    
                except ValueError:
                    print("❌ Valores inválidos")
                except Exception as e:
                    print(f"❌ Error en simulación: {str(e)}")
                        
            elif option == '15':
                print("👋 ¡Gracias por usar el Sistema de Análisis de Satélites!")
                break
                        
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
                        days = int(input("📅 Días hacia el futuro (máx 4): ") or "4")
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
