#!/usr/bin/env python3
"""
API de Integración Web para el Sistema Avanzado de Análisis de Colisiones
Conecta el AdvancedCollisionAnalyzer con la interfaz web Cesium
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import os
import sys

# Importar nuestro sistema de análisis
try:
    from AdvancedCollisionAnalyzer import AdvancedSatelliteAnalyzer
except ImportError:
    print(" Error: No se puede importar AdvancedCollisionAnalyzer")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Permitir requests desde el frontend

# Instancia global del analizador
analyzer = None

@app.route('/')
def index():
    """Servir la página principal"""
    try:
        with open('satellite_collision_viewer.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1> Sistema de Análisis de Colisiones</h1>
        <p> Error: No se encuentra satellite_collision_viewer.html</p>
        <p>Asegúrese de que el archivo HTML esté en el mismo directorio.</p>
        """

@app.route('/api/initialize', methods=['POST'])
def initialize_analyzer():
    """Inicializar el analizador de satélites"""
    global analyzer
    try:
        analyzer = AdvancedSatelliteAnalyzer()
        return jsonify({
            'success': True,
            'message': 'Analizador inicializado correctamente',
            'components': [
                'Propagador orbital con perturbaciones físicas',
                'Analizador de colisiones probabilístico',
                'Cálculo automático de tiempo de maniobra',
                'Modelado de incertidumbre avanzado'
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error inicializando analizador: {e}'
        }), 500

@app.route('/api/load_satellites', methods=['POST'])
def load_satellites():
    """Cargar datos de satélites"""
    global analyzer
    
    if analyzer is None:
        return jsonify({
            'success': False,
            'error': 'Analizador no inicializado'
        }), 400
    
    try:
        success = analyzer.load_satellite_data()
        
        if success:
            satellite_list = []
            for sat_id, satellite in list(analyzer.satellites.items())[:10]:  # Limitar para demo
                try:
                    # Obtener posición actual
                    current_time = analyzer.ts.now()
                    sat_pos = satellite.at(current_time)
                    
                    position = sat_pos.position.km
                    velocity = sat_pos.velocity.km_per_s
                    altitude = float(np.linalg.norm(position) - analyzer.earth_radius)
                    
                    satellite_list.append({
                        'name': sat_id,
                        'position': position.tolist(),
                        'velocity': velocity.tolist(),
                        'altitude': altitude,
                        'radius': 5.0  # metros, valor por defecto
                    })
                except Exception as e:
                    print(f"Error procesando {sat_id}: {e}")
                    continue
            
            return jsonify({
                'success': True,
                'satellites': satellite_list,
                'count': len(satellite_list),
                'total_available': len(analyzer.satellites)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No se pudieron cargar los satélites'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error cargando satélites: {e}'
        }), 500

@app.route('/api/analyze_collision/<target_id>', methods=['POST'])
def analyze_collision(target_id):
    """Analizar colisiones para un satélite específico"""
    global analyzer
    
    if analyzer is None:
        return jsonify({
            'success': False,
            'error': 'Analizador no inicializado'
        }), 400
    
    try:
        # Obtener parámetros
        data = request.get_json() or {}
        safety_multiplier = data.get('safety_multiplier', 3.0)
        time_window_hours = data.get('time_window_hours', 72.0)
        
        # Ejecutar análisis
        results = analyzer.analyze_collision_scenario(
            target_id=target_id,
            time_window_hours=time_window_hours,
            safety_multiplier=safety_multiplier
        )
        
        # Formatear resultados para la web
        web_results = {
            'success': True,
            'target_satellite': results['target_satellite'],
            'analysis_time': results['analysis_time'],
            'summary': results['summary'],
            'collision_risks': [],
            'errors': results['errors']
        }
        
        # Procesar riesgos de colisión
        for risk in results['collision_risks']:
            web_risk = {
                'satellite_id': risk['satellite_id'],
                'risk_level': risk['risk_level'],
                'collision_probability': risk['collision_probability'],
                'distance_km': risk['distance_km'],
                'relative_velocity_km_s': risk['relative_velocity_km_s'],
                'maneuver_analysis': risk['maneuver_analysis'],
                'recommendations': risk['recommendations'],
                'physical_perturbations': {
                    'j2_magnitude': float(np.linalg.norm(risk['physical_perturbations']['j2_accel'])),
                    'drag_magnitude': float(np.linalg.norm(risk['physical_perturbations']['drag_accel'])),
                    'altitude_km': risk['physical_perturbations']['altitude_km']
                }
            }
            web_results['collision_risks'].append(web_risk)
        
        return jsonify(web_results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error en análisis: {e}'
        }), 500

@app.route('/api/calculate_maneuver', methods=['POST'])
def calculate_maneuver():
    """Calcular tiempo de maniobra específico"""
    global analyzer
    
    if analyzer is None:
        return jsonify({
            'success': False,
            'error': 'Analizador no inicializado'
        }), 400
    
    try:
        data = request.get_json()
        collision_data = data.get('collision_data')
        safety_multiplier = data.get('safety_multiplier', 3.0)
        
        if not collision_data:
            return jsonify({
                'success': False,
                'error': 'Datos de colisión requeridos'
            }), 400
        
        # Calcular tiempo de maniobra
        maneuver_result = analyzer.calculate_maneuver_time(
            collision_data, 
            safety_multiplier
        )
        
        return jsonify({
            'success': True,
            'maneuver_analysis': maneuver_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error calculando maniobra: {e}'
        }), 500

@app.route('/api/status')
def status():
    """Estado del sistema"""
    global analyzer
    
    return jsonify({
        'system_status': 'online',
        'analyzer_initialized': analyzer is not None,
        'satellites_loaded': len(analyzer.satellites) if analyzer else 0,
        'version': '1.0.0',
        'capabilities': [
            'Análisis de colisiones avanzado',
            'Cálculo de tiempo de maniobra',
            'Perturbaciones físicas (J2 + arrastre)',
            'Análisis probabilístico',
            'Visualización 3D con Cesium'
        ]
    })

if __name__ == '__main__':
    print(" Iniciando API de Integración Web...")
    print(" Servidor disponible en: http://localhost:5000")
    print(" Sistema integrado con Cesium.js")
    print(" AdvancedCollisionAnalyzer Backend")
    
    # Importar numpy para los cálculos
    import numpy as np
    
    app.run(debug=True, host='0.0.0.0', port=5000)
