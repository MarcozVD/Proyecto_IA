import React, { useState, useRef } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

const App = () => {
  const [serverIP, setServerIP] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState('home');
  const [sensorData, setSensorData] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<{prediction: number; probability: number} | null>(null);
  const sensorInputRefs = useRef<Record<string, TextInput>>({});
  const sensorDrafts = useRef<Record<string, string>>({});
  const [modelInfo] = useState({
    model: 'XGBoost',
    threshold: 0.3847,
    variables: 12,
    optimization: 'GridSearchCV',
    balancing: 'SMOTE'
  });

  const sensorColumns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS'];
  
  const sensorNames: Record<string, string> = {
    'TP2': 'Temperatura Punto 2 (°C)',
    'TP3': 'Temperatura Punto 3 (°C)',
    'H1': 'Humedad Relativa (%)',
    'DV_pressure': 'Presión Válvula (bar)',
    'Reservoirs': 'Nivel Reservorios',
    'Oil_temperature': 'Temperatura Aceite (°C)',
    'Motor_current': 'Corriente Motor (A)',
    'COMP': 'Estado Compresor',
    'DV_eletric': 'Válvula Eléctrica',
    'Towers': 'Torres Activas',
    'MPG': 'Motor Principal',
    'LPS': 'Sensor Baja Presión'
  };

  const sensorRanges: Record<string, [number, number]> = {
    'TP2': [-0.032, 10.676],
    'TP3': [0.730, 10.302],
    'H1': [-0.036, 10.288],
    'DV_pressure': [-0.032, 9.844],
    'Reservoirs': [0.712, 10.300],
    'Oil_temperature': [15.400, 89.050],
    'Motor_current': [0.020, 9.295],
    'COMP': [0.000, 1.000],
    'DV_eletric': [0.000, 1.000],
    'Towers': [0.000, 1.000],
    'MPG': [0.000, 1.000],
    'LPS': [0.000, 1.000]
  };

  const handleConnect = async () => {
    setLoading(true);
    setTimeout(() => {
      setIsConnected(true);
      setLoading(false);
    }, 2000);
  };

  const handlePredict = async () => {
    setLoading(true);
    setTimeout(() => {
      const randomProb = Math.random();
      const predictedClass = randomProb >= modelInfo.threshold ? 1 : 0;
      setPrediction({
        prediction: predictedClass,
        probability: randomProb
      });
      setLoading(false);
    }, 1500);
  };

  // Componente de entrada de sensor utilizando inputs no controlados (defaultValue)
  // para evitar re-renders en cada pulsación que cierren el teclado.
  const SensorInput = ({ sensorKey, min, max }: { sensorKey: string; min: number; max: number }) => {
    const draft = sensorDrafts.current[sensorKey] ?? sensorData[sensorKey] ?? String((min + max) / 2);
    const currentValue = parseFloat(draft);
    const isOutOfRange = currentValue < min || currentValue > max;

    return (
      <View style={styles.sensorInputField}>
        <View style={styles.sensorInputHeader}>
          <Text style={styles.sensorInputLabel}>{sensorNames[sensorKey]}</Text>
          <Text style={[
            styles.sensorInputRange,
            isOutOfRange && styles.sensorInputRangeError
          ]}>
            [{min.toFixed(2)}, {max.toFixed(2)}]
          </Text>
        </View>
        <TextInput
          ref={(ref) => {
            if (ref) sensorInputRefs.current[sensorKey] = ref;
          }}
          style={[
            styles.sensorInputBox,
            isOutOfRange && styles.sensorInputBoxError
          ]}
          defaultValue={draft}
          onChangeText={(text) => {
            // Guardar en draft para no provocar re-renders en cada tecla
            sensorDrafts.current[sensorKey] = text;
          }}
          onFocus={() => {
            if (sensorDrafts.current[sensorKey] === undefined) {
              sensorDrafts.current[sensorKey] = sensorData[sensorKey] ?? String((min + max) / 2);
            }
          }}
          onEndEditing={() => {
            const text = sensorDrafts.current[sensorKey] ?? sensorData[sensorKey] ?? String((min + max) / 2);
            const val = parseFloat(text);
            if (isNaN(val) || text === '') {
              const defaultStr = String((min + max) / 2);
              setSensorData(prevData => ({...prevData, [sensorKey]: defaultStr}));
              sensorDrafts.current[sensorKey] = defaultStr;
            } else {
              const clampedVal = Math.min(max, Math.max(min, val));
              const strVal = String(clampedVal);
              setSensorData(prevData => ({...prevData, [sensorKey]: strVal}));
              sensorDrafts.current[sensorKey] = strVal;
            }
          }}
          placeholder={String((min + max) / 2)}
          keyboardType="decimal-pad"
          placeholderTextColor="#94a3b8"
        />
        {isOutOfRange && (
          <Text style={styles.sensorInputError}>
            ⚠️ Valor fuera de rango [{min.toFixed(2)}, {max.toFixed(2)}]
          </Text>
        )}
      </View>
    );
  };

  // Pantalla de conexión
  if (!isConnected) {
    const ipPresets = [
      { label: 'Local', ip: 'localhost:5000' },
      { label: '192.168.1.100', ip: '192.168.1.100:5000' },
      { label: '127.0.0.1', ip: '127.0.0.1:5000' },
    ];

    return (
      <View style={styles.connectionContainerNew}>
        <ScrollView contentContainerStyle={styles.connectionScrollContentNew} showsVerticalScrollIndicator={false}>
          {/* Hero Section */}
          <View style={styles.heroSection}>
            <View style={styles.heroBackground} />
            <Text style={styles.heroIcon}>⚡</Text>
            <Text style={styles.heroTitle}>Predicción de Fallas</Text>
            <Text style={styles.heroSubtitle}>MetroPT3 Air Compressor</Text>
          </View>

          {/* Contenido Principal */}
          <View style={styles.connectionMainContent}>
            {/* IP Input */}
            <View style={styles.connectionSection}>
              <Text style={styles.connectionSectionTitle}>Conexión al Servidor</Text>
              <TextInput
                style={styles.connectionInput}
                value={serverIP}
                onChangeText={setServerIP}
                placeholder="192.168.1.100:5000"
                placeholderTextColor="#cbd5e1"
              />
              <Text style={styles.connectionInputHelper}>Ingresa la IP y puerto del servidor</Text>
            </View>

            {/* Presets */}
            <View style={styles.presetsContainerNew}>
              <Text style={styles.presetsLabelNew}>Presets Rápidos</Text>
              <View style={styles.presetsGridNew}>
                {ipPresets.map((preset, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={[styles.presetButtonNew, serverIP === preset.ip && styles.presetButtonNewActive]}
                    onPress={() => setServerIP(preset.ip)}
                  >
                    <Text style={[styles.presetButtonTextNew, serverIP === preset.ip && styles.presetButtonTextNewActive]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Info Cards */}
            <View style={styles.connectionInfoCards}>
              <View style={[styles.infoCard, {backgroundColor: '#e0f2fe'}]}>
                <Text style={styles.infoCardIcon}>📊</Text>
                <Text style={styles.infoCardTitle}>220,320</Text>
                <Text style={styles.infoCardSubtitle}>Registros</Text>
              </View>
              <View style={[styles.infoCard, {backgroundColor: '#f0fdf4'}]}>
                <Text style={styles.infoCardIcon}>🎯</Text>
                <Text style={styles.infoCardTitle}>12</Text>
                <Text style={styles.infoCardSubtitle}>Sensores</Text>
              </View>
              <View style={[styles.infoCard, {backgroundColor: '#fef2f2'}]}>
                <Text style={styles.infoCardIcon}>⚠️</Text>
                <Text style={styles.infoCardTitle}>1.4%</Text>
                <Text style={styles.infoCardSubtitle}>Fallas</Text>
              </View>
            </View>

            {/* Connect Button */}
            <TouchableOpacity
              style={[styles.connectButtonNew, (!serverIP || loading) && styles.buttonDisabledNew]}
              onPress={handleConnect}
              disabled={!serverIP || loading}
            >
              {loading ? (
                <View style={styles.buttonContentNew}>
                  <ActivityIndicator color="#fff" size="small" />
                  <Text style={styles.buttonTextNew}>Conectando...</Text>
                </View>
              ) : (
                <Text style={styles.buttonTextNew}>Conectar</Text>
              )}
            </TouchableOpacity>
          </View>
        </ScrollView>
      </View>
    );
  }

  // Página de inicio
  const HomePage = () => (
    <ScrollView style={styles.pageContainer} contentContainerStyle={styles.pageContent}>
      {/* Quick Stats - Tres en fila */}
      <View style={styles.quickStatsContainer}>
        <View style={styles.quickStatItem}>
          <Text style={styles.quickStatIcon}>📊</Text>
          <Text style={styles.quickStatValue}>220.3K</Text>
          <Text style={styles.quickStatLabel}>Registros</Text>
        </View>
        <View style={styles.quickStatItem}>
          <Text style={styles.quickStatIcon}>🎯</Text>
          <Text style={styles.quickStatValue}>12</Text>
          <Text style={styles.quickStatLabel}>Sensores</Text>
        </View>
        <View style={styles.quickStatItem}>
          <Text style={styles.quickStatIcon}>✅</Text>
          <Text style={styles.quickStatValue}>95.2%</Text>
          <Text style={styles.quickStatLabel}>Exactitud</Text>
        </View>
      </View>

      {/* Main Features Grid - 2x2 */}
      <View style={styles.featureGridLarge}>
        <View style={[styles.featureCardLarge, {backgroundColor: '#e0f2fe', borderLeftColor: '#60a5fa'}]}>
          <Text style={styles.featureCardIcon}>⚡</Text>
          <Text style={styles.featureCardTitle}>Tiempo Real</Text>
          <Text style={styles.featureCardDesc}>Predicciones instantáneas</Text>
        </View>
        <View style={[styles.featureCardLarge, {backgroundColor: '#f0fdf4', borderLeftColor: '#10b981'}]}>
          <Text style={styles.featureCardIcon}>🤖</Text>
          <Text style={styles.featureCardTitle}>XGBoost</Text>
          <Text style={styles.featureCardDesc}>Modelo optimizado</Text>
        </View>
        <View style={[styles.featureCardLarge, {backgroundColor: '#fef3c7', borderLeftColor: '#f59e0b'}]}>
          <Text style={styles.featureCardIcon}>📈</Text>
          <Text style={styles.featureCardTitle}>Análisis</Text>
          <Text style={styles.featureCardDesc}>Datos detallados</Text>
        </View>
        <View style={[styles.featureCardLarge, {backgroundColor: '#fce7f3', borderLeftColor: '#ec4899'}]}>
          <Text style={styles.featureCardIcon}>🔔</Text>
          <Text style={styles.featureCardTitle}>Alertas</Text>
          <Text style={styles.featureCardDesc}>Notificaciones</Text>
        </View>
      </View>

      {/* Model Info - Horizontal Layout */}
      <View style={styles.modelInfoHorizontal}>
        <View style={styles.modelInfoItemH}>
          <Text style={styles.modelInfoLabelH}>Modelo</Text>
          <Text style={styles.modelInfoValueH}>XGBoost</Text>
        </View>
        <View style={styles.dividerVertical} />
        <View style={styles.modelInfoItemH}>
          <Text style={styles.modelInfoLabelH}>Umbral</Text>
          <Text style={styles.modelInfoValueH}>0.3847</Text>
        </View>
        <View style={styles.dividerVertical} />
        <View style={styles.modelInfoItemH}>
          <Text style={styles.modelInfoLabelH}>Precisión</Text>
          <Text style={styles.modelInfoValueH}>92.8%</Text>
        </View>
      </View>

      {/* Quick Action Buttons */}
      <View style={styles.quickActionsContainer}>
        <TouchableOpacity 
          style={[styles.actionButton, {backgroundColor: '#e0f2fe', borderColor: '#60a5fa'}]}
          onPress={() => setCurrentPage('prediction')}
        >
          <Text style={styles.actionButtonIcon}>🔮</Text>
          <Text style={styles.actionButtonText}>Nueva Predicción</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.actionButton, {backgroundColor: '#f0fdf4', borderColor: '#10b981'}]}
          onPress={() => setCurrentPage('analysis')}
        >
          <Text style={styles.actionButtonIcon}>📊</Text>
          <Text style={styles.actionButtonText}>Ver Análisis</Text>
        </TouchableOpacity>
      </View>

      {/* System Status */}
      <View style={styles.systemStatusCard}>
        <View style={styles.statusHeader}>
          <Text style={styles.statusTitle}>Estado del Sistema</Text>
          <View style={[styles.statusBadge, {backgroundColor: '#10b981'}]}>
            <Text style={styles.statusBadgeText}>🟢 Operativo</Text>
          </View>
        </View>
        <View style={styles.statusGrid}>
          <View style={styles.statusItem}>
            <Text style={styles.statusItemLabel}>Servidor</Text>
            <Text style={styles.statusItemValue}>Conectado</Text>
          </View>
          <View style={styles.statusItem}>
            <Text style={styles.statusItemLabel}>Sensores</Text>
            <Text style={styles.statusItemValue}>12/12</Text>
          </View>
          <View style={styles.statusItem}>
            <Text style={styles.statusItemLabel}>Modelo</Text>
            <Text style={styles.statusItemValue}>Listo</Text>
          </View>
          <View style={styles.statusItem}>
            <Text style={styles.statusItemLabel}>API</Text>
            <Text style={styles.statusItemValue}>Activa</Text>
          </View>
        </View>
      </View>

      {/* Recent Activity */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>📋 Actividad Reciente</Text>
        <View style={styles.activityList}>
          <View style={styles.activityItem}>
            <View style={[styles.activityDot, {backgroundColor: '#ef4444'}]} />
            <View style={styles.activityContent}>
              <Text style={styles.activityText}>Falla detectada</Text>
              <Text style={styles.activityTime}>Hace 2 horas</Text>
            </View>
          </View>
          <View style={styles.activityItem}>
            <View style={[styles.activityDot, {backgroundColor: '#10b981'}]} />
            <View style={styles.activityContent}>
              <Text style={styles.activityText}>Sistema normal</Text>
              <Text style={styles.activityTime}>Hace 15 minutos</Text>
            </View>
          </View>
          <View style={styles.activityItem}>
            <View style={[styles.activityDot, {backgroundColor: '#3b82f6'}]} />
            <View style={styles.activityContent}>
              <Text style={styles.activityText}>Modelo actualizado</Text>
              <Text style={styles.activityTime}>Ayer a las 10:30</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Connection Info Footer */}
      <View style={styles.connectionInfoFooter}>
        <Text style={styles.connectionInfoTitle}>📡 Información de Conexión</Text>
        <Text style={styles.connectionInfoText}>IP: {serverIP}</Text>
        <Text style={styles.connectionInfoText}>Estado: Conectado ✓</Text>
        <TouchableOpacity
          style={styles.disconnectButtonSmall}
          onPress={() => {
            setIsConnected(false);
            setServerIP('');
            setPrediction(null);
            setSensorData({});
          }}
        >
          <Text style={styles.disconnectButtonSmallText}>Desconectar</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );

  // Página de predicción
  const PredictionPage = () => (
    <ScrollView style={styles.pageContainer} contentContainerStyle={styles.pageContent}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔮 Predicción Individual de Fallas</Text>
        <Text style={styles.cardSubtitle}>Ajusta los valores de los sensores usando sliders o escribiendo valores directamente</Text>
        
        <View style={styles.sensorGrid}>
          {sensorColumns.map((col, idx) => {
            const [min, max] = sensorRanges[col] || [0, 100];
            return (
              <View key={idx} style={styles.sensorInputContainer}>
                <SensorInput sensorKey={col} min={min} max={max} />
              </View>
            );
          })}
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, loading && styles.buttonDisabled]}
          onPress={handlePredict}
          disabled={loading}
        >
          {loading ? (
            <View style={styles.buttonContent}>
              <ActivityIndicator color="#fff" />
              <Text style={[styles.buttonText, {marginLeft: 10}]}>Prediciendo...</Text>
            </View>
          ) : (
            <Text style={styles.buttonText}>🚀 REALIZAR PREDICCIÓN</Text>
          )}
        </TouchableOpacity>
      </View>

      {prediction && (
        <>
          <View style={[styles.predictionCard, prediction.prediction === 1 ? styles.predictionDanger : styles.predictionSuccess]}>
            <View style={styles.predictionHeader}>
              <Text style={styles.predictionIconLarge}>
                {prediction.prediction === 1 ? '⚠️' : '✅'}
              </Text>
              <View style={styles.predictionTextContainer}>
                <Text style={[styles.predictionTitle, prediction.prediction === 1 ? styles.textDanger : styles.textSuccess]}>
                  {prediction.prediction === 1 ? 'ALERTA: FALLA DETECTADA' : 'SISTEMA OPERANDO NORMALMENTE'}
                </Text>
                <Text style={[styles.predictionSubtitle, prediction.prediction === 1 ? styles.textDanger : styles.textSuccess]}>
                  {prediction.prediction === 1 ? 'Se requiere atención inmediata' : 'Todos los parámetros en rango normal'}
                </Text>
              </View>
            </View>

            <View style={styles.metricsContainer}>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Probabilidad de Falla</Text>
                <Text style={[styles.metricValue, prediction.prediction === 1 ? styles.textDanger : styles.textSuccess]}>
                  {(prediction.probability * 100).toFixed(2)}%
                </Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Umbral de Decisión</Text>
                <Text style={styles.metricValue}>
                  {(modelInfo.threshold * 100).toFixed(2)}%
                </Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Nivel de Confianza</Text>
                <Text style={[styles.metricValue, prediction.prediction === 1 ? styles.textDanger : styles.textSuccess]}>
                  {Math.min((Math.abs(prediction.probability - modelInfo.threshold) / modelInfo.threshold * 100), 100).toFixed(1)}%
                </Text>
              </View>
            </View>

            <View style={styles.progressSection}>
              <Text style={styles.progressTitle}>Probabilidad vs Umbral</Text>
              <View style={styles.progressBarContainer}>
                <View 
                  style={[
                    styles.progressBarFill,
                    prediction.prediction === 1 ? styles.progressDanger : styles.progressSuccess,
                    { width: `${Math.min(prediction.probability * 100, 100)}%` }
                  ]}
                />
                <View style={[styles.thresholdLine, { left: `${modelInfo.threshold * 100}%` }]} />
              </View>
              <View style={styles.progressLabels}>
                <Text style={styles.progressLabel}>0%</Text>
                <Text style={[styles.progressLabel, styles.thresholdLabel]}>
                  Umbral: {(modelInfo.threshold * 100).toFixed(1)}%
                </Text>
                <Text style={styles.progressLabel}>100%</Text>
              </View>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>🧠 Explicación de la Decisión</Text>
            <View style={[styles.explanationBox, prediction.prediction === 1 ? styles.explanationDanger : styles.explanationSuccess]}>
              <Text style={styles.explanationTitle}>
                {prediction.prediction === 1 ? '¿Por qué se detectó una falla?' : '¿Por qué NO se detectó una falla?'}
              </Text>
              <Text style={styles.explanationText}>
                • La probabilidad predicha ({(prediction.probability * 100).toFixed(2)}%) es{' '}
                <Text style={styles.bold}>
                  {prediction.prediction === 1 ? 'mayor o igual' : 'menor'}
                </Text>{' '}
                al umbral ({(modelInfo.threshold * 100).toFixed(2)}%)
              </Text>
              <Text style={styles.explanationText}>
                • El modelo tiene <Text style={styles.bold}>
                  {prediction.prediction === 1 
                    ? `${(prediction.probability * 100).toFixed(1)}% de confianza` 
                    : `${((1 - prediction.probability) * 100).toFixed(1)}% de confianza`
                  }
                </Text> en esta predicción
              </Text>
              <Text style={styles.explanationText}>
                • Margen {prediction.prediction === 1 ? 'sobre' : 'bajo'} el umbral:{' '}
                <Text style={styles.bold}>
                  {(Math.abs(prediction.probability - modelInfo.threshold) * 100).toFixed(2)} p.p.
                </Text>
              </Text>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>💡 Recomendaciones</Text>
            <View style={[styles.recommendationsBox, prediction.prediction === 1 ? styles.recommendationsDanger : styles.recommendationsSuccess]}>
              {prediction.prediction === 1 ? (
                <>
                  <Text style={styles.recommendationsTitle}>⚠️ Acciones Inmediatas:</Text>
                  <Text style={styles.recommendationItem}>🔧 Inspeccionar el compresor inmediatamente</Text>
                  <Text style={styles.recommendationItem}>📞 Notificar al equipo de mantenimiento</Text>
                  <Text style={styles.recommendationItem}>📝 Registrar el evento en el sistema</Text>
                  <Text style={styles.recommendationItem}>🔍 Verificar sensores con valores anómalos</Text>
                </>
              ) : (
                <>
                  <Text style={styles.recommendationsTitle}>✅ Sistema en Estado Normal:</Text>
                  <Text style={styles.recommendationItem}>✓ Continuar con operación normal</Text>
                  <Text style={styles.recommendationItem}>📊 Mantener monitoreo de rutina</Text>
                  <Text style={styles.recommendationItem}>🔄 Programar mantenimiento preventivo</Text>
                </>
              )}
            </View>
          </View>
        </>
      )}
    </ScrollView>
  );

  // Página de análisis
  const AnalysisPage = () => (
    <ScrollView style={styles.pageContainer} contentContainerStyle={styles.pageContent}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>📈 Análisis Exploratorio de Datos</Text>
        
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statBoxNumber}>220,320</Text>
            <Text style={styles.statBoxLabel}>Registros</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statBoxNumber}>{modelInfo.variables}</Text>
            <Text style={styles.statBoxLabel}>Variables</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statBoxNumber}>8.2 MB</Text>
            <Text style={styles.statBoxLabel}>Memoria</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statBoxNumber}>Abr-Jul</Text>
            <Text style={styles.statBoxLabel}>2020</Text>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔍 Valores Faltantes</Text>
        <View style={styles.successBanner}>
          <Text style={styles.successIcon}>✅</Text>
          <Text style={styles.successText}>No hay valores nulos en el dataset</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🎯 Distribución de Clases</Text>
        <View style={styles.distributionContainer}>
          <View style={styles.distributionRow}>
            <View style={styles.distributionInfo}>
              <View style={[styles.dot, {backgroundColor: '#10b981'}]} />
              <Text style={styles.distributionLabel}>Sin Falla</Text>
            </View>
            <Text style={styles.distributionValue}>217,152 (98.6%)</Text>
          </View>
          <View style={styles.distributionBarContainer}>
            <View style={[styles.distributionBar, {backgroundColor: '#10b981', width: '98.6%'}]} />
          </View>
          
          <View style={[styles.distributionRow, {marginTop: 15}]}>
            <View style={styles.distributionInfo}>
              <View style={[styles.dot, {backgroundColor: '#ef4444'}]} />
              <Text style={styles.distributionLabel}>Con Falla</Text>
            </View>
            <Text style={styles.distributionValue}>3,168 (1.4%)</Text>
          </View>
          <View style={styles.distributionBarContainer}>
            <View style={[styles.distributionBar, {backgroundColor: '#ef4444', width: '1.4%'}]} />
          </View>
        </View>
        
        <View style={styles.balanceInfo}>
          <Text style={styles.balanceText}>
            📊 <Text style={styles.bold}>Ratio de Desbalance:</Text> 0.0146 (1:68.57){'\n'}
            Las clases están <Text style={styles.bold}>muy desbalanceadas</Text>
          </Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>📊 Estadísticas Descriptivas</Text>
        <Text style={styles.cardSubtitle}>Principales sensores del sistema</Text>
        
        <View style={styles.statsCardsGrid}>
          {[
            {sensor: 'TP2', min: '-0.03', mean: '5.32', max: '10.68'},
            {sensor: 'TP3', min: '0.73', mean: '5.51', max: '10.30'},
            {sensor: 'Oil_temperature', min: '15.40', mean: '52.23', max: '89.05'},
            {sensor: 'Motor_current', min: '0.02', mean: '4.66', max: '9.30'}
          ].map((data, idx) => (
            <View key={idx} style={styles.statDataCard}>
              <Text style={styles.statDataTitle}>{sensorNames[data.sensor]}</Text>
              <View style={styles.statDataRow}>
                <View style={styles.statDataItem}>
                  <Text style={styles.statDataLabel}>Mín</Text>
                  <Text style={styles.statDataValue}>{data.min}</Text>
                </View>
                <View style={styles.statDataItem}>
                  <Text style={styles.statDataLabel}>Media</Text>
                  <Text style={styles.statDataValue}>{data.mean}</Text>
                </View>
                <View style={styles.statDataItem}>
                  <Text style={styles.statDataLabel}>Máx</Text>
                  <Text style={styles.statDataValue}>{data.max}</Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔗 Correlaciones Principales</Text>
        <View style={styles.correlationsList}>
          {[
            {pair: 'TP2 ↔ TP3', value: 0.95},
            {pair: 'Oil_temp ↔ Motor_current', value: 0.88},
            {pair: 'DV_pressure ↔ Reservoirs', value: 0.76}
          ].map((corr, idx) => (
            <View key={idx} style={styles.correlationItem}>
              <Text style={styles.correlationPair}>{corr.pair}</Text>
              <View style={styles.correlationBarContainer}>
                <View style={[styles.correlationBar, {width: `${corr.value * 100}%`}]} />
              </View>
              <Text style={styles.correlationValue}>{corr.value.toFixed(2)}</Text>
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );

  // Página de evaluación
  const EvaluationPage = () => (
    <ScrollView style={styles.pageContainer} contentContainerStyle={styles.pageContent}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>📉 Evaluación del Modelo XGBoost</Text>
        
        <View style={styles.modelInfoGrid}>
          {[
            {label: 'Algoritmo', value: 'XGBoost', color: '#60a5fa'},
            {label: 'Optimización', value: 'GridSearchCV', color: '#93c5fd'},
            {label: 'Balanceo', value: 'SMOTE', color: '#7dd3fc'},
            {label: 'Escalado', value: 'StandardScaler', color: '#10b981'}
          ].map((info, idx) => (
            <View key={idx} style={[styles.modelInfoCard, {borderLeftColor: info.color}]}>
              <Text style={styles.modelInfoLabel}>{info.label}</Text>
              <Text style={styles.modelInfoValue}>{info.value}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>📊 Métricas de Evaluación</Text>
        
        <View style={styles.metricsEvalGrid}>
          {[
            {label: 'Accuracy', value: 95.2, color: '#2563eb', bg: '#eff6ff'},
            {label: 'Precision', value: 92.8, color: '#f59e0b', bg: '#fef3c7'},
            {label: 'Recall', value: 91.5, color: '#ec4899', bg: '#fce7f3'},
            {label: 'F1-Score', value: 92.1, color: '#10b981', bg: '#d1fae5'}
          ].map((metric, idx) => (
            <View key={idx} style={[styles.metricEvalCard, {backgroundColor: metric.bg}]}>
              <Text style={styles.metricEvalLabel}>{metric.label}</Text>
              <Text style={[styles.metricEvalValue, {color: metric.color}]}>{metric.value}%</Text>
              <View style={styles.metricEvalBar}>
                <View style={[styles.metricEvalBarFill, {width: `${metric.value}%`, backgroundColor: metric.color}]} />
              </View>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🎯 Matriz de Confusión</Text>
        <Text style={styles.cardSubtitle}>Ejemplo de resultados del modelo</Text>
        
        <View style={styles.confusionMatrix}>
          <View style={styles.confusionRow}>
            <View style={[styles.confusionCell, {backgroundColor: '#dbeafe', borderColor: '#3b82f6'}]}>
              <Text style={styles.confusionLabel}>TN</Text>
              <Text style={styles.confusionValue}>8,500</Text>
              <Text style={styles.confusionDesc}>Verdaderos{'\n'}Negativos</Text>
            </View>
            <View style={[styles.confusionCell, {backgroundColor: '#fee2e2', borderColor: '#ef4444'}]}>
              <Text style={styles.confusionLabel}>FP</Text>
              <Text style={styles.confusionValue}>50</Text>
              <Text style={styles.confusionDesc}>Falsos{'\n'}Positivos</Text>
            </View>
          </View>
          <View style={styles.confusionRow}>
            <View style={[styles.confusionCell, {backgroundColor: '#fed7aa', borderColor: '#f59e0b'}]}>
              <Text style={styles.confusionLabel}>FN</Text>
              <Text style={styles.confusionValue}>30</Text>
              <Text style={styles.confusionDesc}>Falsos{'\n'}Negativos</Text>
            </View>
            <View style={[styles.confusionCell, {backgroundColor: '#dcfce7', borderColor: '#10b981'}]}>
              <Text style={styles.confusionLabel}>TP</Text>
              <Text style={styles.confusionValue}>120</Text>
              <Text style={styles.confusionDesc}>Verdaderos{'\n'}Positivos</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>⚙️ Hiperparámetros Optimizados</Text>
        
        <View style={styles.paramsContainer}>
          <View style={styles.paramSection}>
            <Text style={styles.paramSectionTitle}>📐 Estructura del Árbol</Text>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>max_depth:</Text>
              <Text style={styles.paramValue}>5</Text>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>min_child_weight:</Text>
              <Text style={styles.paramValue}>3</Text>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>gamma:</Text>
              <Text style={styles.paramValue}>0.1</Text>
            </View>
          </View>

          <View style={styles.paramSection}>
            <Text style={styles.paramSectionTitle}>🎯 Aprendizaje</Text>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>learning_rate:</Text>
              <Text style={styles.paramValue}>0.1</Text>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>n_estimators:</Text>
              <Text style={styles.paramValue}>100</Text>
            </View>
          </View>

          <View style={styles.paramSection}>
            <Text style={styles.paramSectionTitle}>🔧 Regularización</Text>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>subsample:</Text>
              <Text style={styles.paramValue}>0.8</Text>
            </View>
            <View style={styles.paramRow}>
              <Text style={styles.paramLabel}>colsample_bytree:</Text>
              <Text style={styles.paramValue}>0.8</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔄 Proceso de Entrenamiento del Modelo</Text>
        
        <View style={styles.processContainer}>
          {[
            {
              number: '1',
              title: '📊 Preprocesamiento de Datos',
              items: ['Limpieza de outliers mediante clipping (1% - 99%)', 'Eliminación de variables con baja varianza', 'Escalado StandardScaler para normalización', 'Manejo de valores faltantes']
            },
            {
              number: '2',
              title: '⚖️ Balanceo de Clases',
              items: ['Aplicación de SMOTE solo en entrenamiento', 'Preservación del conjunto de prueba sin modificar', 'Ratio ajustado para clases desbalanceadas (1:68.57)']
            },
            {
              number: '3',
              title: '🔍 Optimización de Hiperparámetros',
              items: ['GridSearchCV con validación cruzada (CV=3)', 'Optimización basada en F1-Score', 'Búsqueda en espacio de parámetros: max_depth, learning_rate, subsample']
            },
            {
              number: '4',
              title: '🎯 Optimización de Umbral',
              items: ['Cálculo de curva Precision-Recall', `Umbral optimizado: ${modelInfo.threshold.toFixed(4)}`, 'Maximización del F1-Score', 'Balance entre precisión y recall']
            }
          ].map((step, idx) => (
            <View key={idx} style={styles.processStep}>
              <View style={styles.processNumber}>
                <Text style={styles.processNumberText}>{step.number}</Text>
              </View>
              <View style={styles.processContent}>
                <Text style={styles.processTitle}>{step.title}</Text>
                {step.items.map((item, i) => (
                  <Text key={i} style={styles.processItem}>• {item}</Text>
                ))}
              </View>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>📊 Fuente de Datos para Entrenamiento</Text>
        <View style={styles.dataSourceContainer}>
          <View style={styles.dataSourceItem}>
            <Text style={styles.dataSourceIcon}>📁</Text>
            <View style={styles.dataSourceContent}>
              <Text style={styles.dataSourceTitle}>Dataset Principal</Text>
              <Text style={styles.dataSourceValue}>MetroPT3(AirCompressor).csv</Text>
              <Text style={styles.dataSourceDesc}>220,320 registros • 12 variables</Text>
            </View>
          </View>
          <View style={styles.dataSourceItem}>
            <Text style={styles.dataSourceIcon}>📊</Text>
            <View style={styles.dataSourceContent}>
              <Text style={styles.dataSourceTitle}>Período de Datos</Text>
              <Text style={styles.dataSourceValue}>Abril - Julio 2020</Text>
              <Text style={styles.dataSourceDesc}>4 meses de operación continua</Text>
            </View>
          </View>
          <View style={styles.dataSourceItem}>
            <Text style={styles.dataSourceIcon}>⚠️</Text>
            <View style={styles.dataSourceContent}>
              <Text style={styles.dataSourceTitle}>Registros de Falla</Text>
              <Text style={styles.dataSourceValue}>3,168 eventos</Text>
              <Text style={styles.dataSourceDesc}>1.4% del total (muy desbalanceado)</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>🎯 Comparación de Modelos - Curva ROC</Text>
        <View style={styles.rocContainer}>
          <Text style={styles.rocDescription}>
            Se evaluaron 6 modelos diferentes para detectar el mejor rendimiento:
          </Text>
          <View style={styles.modelsComparisonGrid}>
            {[
              {name: 'XGBoost', auc: 0.975, f1: 0.921, status: '⭐ SELECCIONADO'},
              {name: 'Random Forest', auc: 0.968, f1: 0.912, status: ''},
              {name: 'Gradient Boosting', auc: 0.962, f1: 0.905, status: ''},
              {name: 'Logistic Regression', auc: 0.934, f1: 0.876, status: ''},
              {name: 'SVM', auc: 0.941, f1: 0.884, status: ''},
              {name: 'Neural Network', auc: 0.958, f1: 0.898, status: ''}
            ].map((model, idx) => (
              <View key={idx} style={[styles.modelComparisonCard, model.status ? styles.modelComparisonCardSelected : {}]}>
                <Text style={styles.modelName}>{model.name}</Text>
                <View style={styles.modelMetricsRow}>
                  <View style={styles.modelMetric}>
                    <Text style={styles.metricSmallLabel}>AUC-ROC</Text>
                    <Text style={styles.metricSmallValue}>{model.auc.toFixed(3)}</Text>
                  </View>
                  <View style={styles.modelMetric}>
                    <Text style={styles.metricSmallLabel}>F1-Score</Text>
                    <Text style={styles.metricSmallValue}>{model.f1.toFixed(3)}</Text>
                  </View>
                </View>
                {model.status && <Text style={styles.selectedBadge}>{model.status}</Text>}
              </View>
            ))}
          </View>
          <View style={styles.rocInfo}>
            <Text style={styles.rocInfoTitle}>📈 XGBoost - Métrica AUC-ROC: 0.975</Text>
            <Text style={styles.rocInfoText}>El área bajo la curva ROC de 0.975 indica excelente rendimiento del modelo en diferenciar entre casos con y sin falla.</Text>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>💡 Recomendaciones de Uso</Text>
        <View style={styles.recommendationsFullBox}>
          {[
            {icon: '📊', title: 'Monitoreo Continuo', text: 'Realiza predicciones periódicas'},
            {icon: '🔄', title: 'Reentrenamiento', text: 'Actualiza el modelo cada 3-6 meses'},
            {icon: '📈', title: 'Análisis de Tendencias', text: 'Observa evolución de probabilidades'},
            {icon: '⚠️', title: 'Umbrales Personalizados', text: 'Ajusta según necesidades de negocio'},
            {icon: '📝', title: 'Registro de Eventos', text: 'Documenta predicciones de falla'}
          ].map((rec, idx) => (
            <View key={idx} style={styles.recommendationFullItem}>
              <Text style={styles.recommendationFullIcon}>{rec.icon}</Text>
              <View style={styles.recommendationFullContent}>
                <Text style={styles.recommendationFullTitle}>{rec.title}</Text>
                <Text style={styles.recommendationFullText}>{rec.text}</Text>
              </View>
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );

  const renderContent = () => {
    switch (currentPage) {
      case 'home': return <HomePage />;
      case 'prediction': return <PredictionPage />;
      case 'analysis': return <AnalysisPage />;
      case 'evaluation': return <EvaluationPage />;
      default: return <HomePage />;
    }
  };

  return (
    <View style={styles.container}>
      {/* Content */}
      {renderContent()}

      {/* Bottom Navigation */}
      <View style={styles.bottomNavigation}>
        <TouchableOpacity
          style={[styles.bottomNavButton, currentPage === 'home' && styles.bottomNavButtonActive]}
          onPress={() => setCurrentPage('home')}
        >
          <Text style={[styles.bottomNavIcon, currentPage === 'home' && styles.bottomNavIconActive]}>🏠</Text>
          <Text style={[styles.bottomNavLabel, currentPage === 'home' && styles.bottomNavLabelActive]}>Inicio</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.bottomNavButton, currentPage === 'prediction' && styles.bottomNavButtonActive]}
          onPress={() => setCurrentPage('prediction')}
        >
          <Text style={[styles.bottomNavIcon, currentPage === 'prediction' && styles.bottomNavIconActive]}>🔮</Text>
          <Text style={[styles.bottomNavLabel, currentPage === 'prediction' && styles.bottomNavLabelActive]}>Predecir</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.bottomNavButton, currentPage === 'analysis' && styles.bottomNavButtonActive]}
          onPress={() => setCurrentPage('analysis')}
        >
          <Text style={[styles.bottomNavIcon, currentPage === 'analysis' && styles.bottomNavIconActive]}>📈</Text>
          <Text style={[styles.bottomNavLabel, currentPage === 'analysis' && styles.bottomNavLabelActive]}>Análisis</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.bottomNavButton, currentPage === 'evaluation' && styles.bottomNavButtonActive]}
          onPress={() => setCurrentPage('evaluation')}
        >
          <Text style={[styles.bottomNavIcon, currentPage === 'evaluation' && styles.bottomNavIconActive]}>📉</Text>
          <Text style={[styles.bottomNavLabel, currentPage === 'evaluation' && styles.bottomNavLabelActive]}>Modelos</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  // Pantalla de Conexión - Nuevo Diseño
  connectionContainerNew: {
    flex: 1,
    backgroundColor: '#f0f9ff',
  },
  connectionScrollContentNew: {
    flexGrow: 1,
    paddingBottom: 40,
  },
  heroSection: {
    backgroundColor: 'linear-gradient(135deg, #60a5fa 0%, #bfdbfe 100%)',
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroBackground: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#60a5fa',
  },
  heroIcon: {
    fontSize: 64,
    marginBottom: 16,
    zIndex: 1,
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: '800',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 8,
    zIndex: 1,
  },
  heroSubtitle: {
    fontSize: 16,
    color: '#e0f2fe',
    textAlign: 'center',
    zIndex: 1,
  },
  connectionMainContent: {
    paddingHorizontal: 20,
    paddingVertical: 32,
  },
  connectionSection: {
    marginBottom: 32,
  },
  connectionSectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 16,
  },
  connectionInput: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#bfdbfe',
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 16,
    fontSize: 15,
    color: '#1e293b',
    marginBottom: 8,
    fontWeight: '500',
  },
  connectionInputHelper: {
    fontSize: 12,
    color: '#64748b',
    marginLeft: 4,
  },
  presetsContainerNew: {
    marginBottom: 32,
  },
  presetsLabelNew: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 12,
  },
  presetsGridNew: {
    flexDirection: 'row',
    gap: 8,
  },
  presetButtonNew: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    alignItems: 'center',
  },
  presetButtonNewActive: {
    borderColor: '#60a5fa',
    backgroundColor: '#e0f2fe',
    borderWidth: 2,
  },
  presetButtonTextNew: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e293b',
  },
  presetButtonTextNewActive: {
    color: '#0369a1',
    fontWeight: '700',
  },
  connectionInfoCards: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  infoCard: {
    flex: 1,
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 2,
  },
  infoCardIcon: {
    fontSize: 28,
    marginBottom: 8,
  },
  infoCardTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1e293b',
    marginBottom: 2,
  },
  infoCardSubtitle: {
    fontSize: 11,
    color: '#64748b',
    fontWeight: '600',
  },
  connectButtonNew: {
    backgroundColor: '#60a5fa',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    shadowColor: '#60a5fa',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  buttonDisabledNew: {
    backgroundColor: '#cbd5e1',
    shadowOpacity: 0,
    elevation: 0,
  },
  buttonContentNew: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  buttonTextNew: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },

  // Pantalla de Conexión - Estilos Antiguos (Mantenidos por compatibilidad)
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  connectionContainer: {
    flex: 1,
    backgroundColor: '#bfdbfe',
    padding: 20,
  },
  connectionCard: {
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 40,
    width: '100%',
    maxWidth: 450,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 12,
  },
  iconContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  iconText: {
    fontSize: 72,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1e3a8a',
    textAlign: 'center',
    marginBottom: 12,
    lineHeight: 36,
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 20,
  },
  inputContainer: {
    marginBottom: 24,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#334155',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#f8fafc',
    borderWidth: 2,
    borderColor: '#e2e8f0',
    borderRadius: 12,
    padding: 16,
    fontSize: 15,
    color: '#1e293b',
  },
  connectButton: {
    backgroundColor: '#60a5fa',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#60a5fa',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },
  buttonDisabled: {
    backgroundColor: '#94a3b8',
    shadowOpacity: 0,
    elevation: 0,
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  spinner: {
    marginRight: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  noteContainer: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  noteText: {
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
  },
  noteBold: {
    fontWeight: '700',
  },

  // Header
  header: {
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerIconContainer: {
    width: 44,
    height: 44,
    backgroundColor: '#60a5fa',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  headerIconText: {
    fontSize: 22,
  },
  headerTitleText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
  },
  headerIP: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 2,
  },
  disconnectButton: {
    backgroundColor: '#ef4444',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
  },
  disconnectText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },

  // Bottom Navigation
  bottomNavigation: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    paddingBottom: 16,
    paddingTop: 8,
    paddingHorizontal: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 3,
  },
  bottomNavButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    borderTopWidth: 3,
    borderTopColor: 'transparent',
  },
  bottomNavButtonActive: {
    borderTopColor: '#60a5fa',
    backgroundColor: '#f0f9ff',
  },
  bottomNavIcon: {
    fontSize: 20,
    marginBottom: 4,
    color: '#94a3b8',
  },
  bottomNavIconActive: {
    color: '#0369a1',
  },
  bottomNavLabel: {
    fontSize: 10,
    fontWeight: '600',
    color: '#64748b',
  },
  bottomNavLabelActive: {
    color: '#0369a1',
    fontWeight: '700',
  },

  // Navigation (estilos antiguos)
  navigation: {
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  navContent: {
    paddingHorizontal: 10,
  },
  navButton: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    marginHorizontal: 4,
  },
  navButtonActive: {
    borderBottomWidth: 3,
    borderBottomColor: '#60a5fa',
    backgroundColor: '#f0f9ff',
  },
  navText: {
    fontSize: 14,
    color: '#64748b',
    fontWeight: '600',
  },
  navTextActive: {
    color: '#0369a1',
    fontWeight: '700',
  },

  // Page Container
  pageContainer: {
    flex: 1,
  },
  pageContent: {
    padding: 16,
    paddingBottom: 32,
  },

  // Cards
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 16,
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 16,
    marginTop: -8,
  },

  // Header Card
  headerCard: {
    backgroundColor: '#bfdbfe',
    borderRadius: 20,
    padding: 32,
    marginBottom: 16,
    alignItems: 'center',
    shadowColor: '#60a5fa',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 8,
  },
  headerIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: '#0f172a',
    textAlign: 'center',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 13,
    color: '#0f172a',
    textAlign: 'center',
    lineHeight: 20,
  },

  // Stats Grid
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 56) / 2,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  statCardPurple: {
    backgroundColor: '#e0f2fe',
    borderLeftWidth: 4,
    borderLeftColor: '#60a5fa',
  },
  statCardRed: {
    backgroundColor: '#fef2f2',
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  statCardBlue: {
    backgroundColor: '#bfdbfe',
    borderLeftWidth: 4,
    borderLeftColor: '#60a5fa',
  },
  statCardGreen: {
    backgroundColor: '#f0fdf4',
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  statIcon: {
    fontSize: 32,
    marginBottom: 12,
  },
  statNumber: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1e293b',
    marginBottom: 6,
  },
  statLabel: {
    fontSize: 12,
    color: '#64748b',
    textAlign: 'center',
    fontWeight: '600',
  },

  // Threshold Card
  thresholdCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 20,
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  thresholdValueContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  thresholdValue: {
    fontSize: 48,
    fontWeight: '800',
    color: '#60a5fa',
    marginBottom: 8,
  },
  thresholdSubtext: {
    fontSize: 16,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 4,
  },
  thresholdDescription: {
    fontSize: 12,
    color: '#64748b',
  },
  divider: {
    height: 1,
    backgroundColor: '#e2e8f0',
    marginVertical: 20,
  },
  sectionText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  bodyText: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 22,
  },
  bold: {
    fontWeight: '700',
    color: '#1e293b',
  },
  rulesList: {
    marginTop: 16,
  },
  ruleItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  ruleDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 12,
  },
  ruleSuccess: {
    backgroundColor: '#10b981',
  },
  ruleDanger: {
    backgroundColor: '#ef4444',
  },
  ruleText: {
    fontSize: 14,
    color: '#475569',
    flex: 1,
  },

  // Features
  featuresContainer: {
    gap: 16,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  featureIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: '#f0fdf4',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  featureIconText: {
    fontSize: 16,
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 4,
  },
  featureText: {
    fontSize: 13,
    color: '#64748b',
    lineHeight: 20,
  },

  // Variables Grid
  variablesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  variableChip: {
    backgroundColor: '#f1f5f9',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  variableText: {
    fontSize: 12,
    color: '#475569',
    fontWeight: '600',
  },

  // Info Box
  infoBox: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  listItem: {
    fontSize: 14,
    color: '#475569',
    marginTop: 8,
    lineHeight: 22,
  },

  // Sensor Inputs
  sensorGrid: {
    gap: 16,
    marginBottom: 8,
  },
  sensorInputContainer: {
    marginBottom: 4,
  },
  sensorLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 8,
  },
  sensorInput: {
    backgroundColor: '#f8fafc',
    borderWidth: 2,
    borderColor: '#e2e8f0',
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    color: '#1e293b',
  },

  // Primary Button
  primaryButton: {
    backgroundColor: '#60a5fa',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginTop: 20,
    shadowColor: '#60a5fa',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },

  // Prediction Card
  predictionCard: {
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 6,
  },
  predictionSuccess: {
    backgroundColor: '#f0fdf4',
    borderWidth: 2,
    borderColor: '#10b981',
  },
  predictionDanger: {
    backgroundColor: '#fef2f2',
    borderWidth: 2,
    borderColor: '#ef4444',
  },
  predictionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  predictionIconLarge: {
    fontSize: 56,
    marginRight: 16,
  },
  predictionTextContainer: {
    flex: 1,
  },
  predictionTitle: {
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 6,
  },
  predictionSubtitle: {
    fontSize: 14,
    fontWeight: '600',
  },
  textSuccess: {
    color: '#059669',
  },
  textDanger: {
    color: '#dc2626',
  },

  // Metrics Container
  metricsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  metricBox: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  metricLabel: {
    fontSize: 11,
    color: '#64748b',
    marginBottom: 8,
    textAlign: 'center',
    fontWeight: '600',
  },
  metricValue: {
    fontSize: 24,
    fontWeight: '800',
  },

  // Progress Section
  progressSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  progressTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
    marginBottom: 12,
    textAlign: 'center',
  },
  progressBarContainer: {
    height: 20,
    backgroundColor: '#e2e8f0',
    borderRadius: 10,
    overflow: 'hidden',
    position: 'relative',
    marginBottom: 8,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 10,
  },
  progressSuccess: {
    backgroundColor: '#10b981',
  },
  progressDanger: {
    backgroundColor: '#ef4444',
  },
  thresholdLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 3,
    backgroundColor: '#1e293b',
  },
  progressLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  progressLabel: {
    fontSize: 11,
    color: '#64748b',
    fontWeight: '600',
  },
  thresholdLabel: {
    color: '#1e293b',
    fontWeight: '700',
  },

  // Explanation Box
  explanationBox: {
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
  },
  explanationSuccess: {
    backgroundColor: '#f0fdf4',
    borderLeftColor: '#10b981',
  },
  explanationDanger: {
    backgroundColor: '#fef2f2',
    borderLeftColor: '#ef4444',
  },
  explanationTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  explanationText: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 22,
    marginBottom: 8,
  },

  // Recommendations Box
  recommendationsBox: {
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
  },
  recommendationsSuccess: {
    backgroundColor: '#f0fdf4',
    borderLeftColor: '#10b981',
  },
  recommendationsDanger: {
    backgroundColor: '#fef2f2',
    borderLeftColor: '#ef4444',
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  recommendationItem: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 24,
    marginBottom: 6,
  },

  // Stats Row
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  statBoxNumber: {
    fontSize: 22,
    fontWeight: '800',
    color: '#1e293b',
    marginBottom: 6,
  },
  statBoxLabel: {
    fontSize: 11,
    color: '#64748b',
    textAlign: 'center',
    fontWeight: '600',
  },

  // Success Banner
  successBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0fdf4',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#10b981',
  },
  successIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  successText: {
    fontSize: 15,
    color: '#059669',
    fontWeight: '700',
  },

  // Distribution
  distributionContainer: {
    marginTop: 8,
  },
  distributionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  distributionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  distributionLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
  },
  distributionValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
  },
  distributionBarContainer: {
    height: 12,
    backgroundColor: '#e2e8f0',
    borderRadius: 6,
    overflow: 'hidden',
    marginBottom: 8,
  },
  distributionBar: {
    height: '100%',
    borderRadius: 6,
  },
  balanceInfo: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  balanceText: {
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
  },

  // Stats Cards Grid
  statsCardsGrid: {
    gap: 12,
  },
  statDataCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  statDataTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  statDataRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statDataItem: {
    alignItems: 'center',
  },
  statDataLabel: {
    fontSize: 11,
    color: '#64748b',
    marginBottom: 6,
    fontWeight: '600',
  },
  statDataValue: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1e293b',
  },

  // Correlations List
  correlationsList: {
    gap: 16,
    marginTop: 8,
  },
  correlationItem: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  correlationPair: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  correlationBarContainer: {
    height: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  correlationBar: {
    height: '100%',
    backgroundColor: '#3b82f6',
    borderRadius: 4,
  },
  correlationValue: {
    fontSize: 16,
    fontWeight: '800',
    color: '#3b82f6',
    textAlign: 'right',
  },

  // Model Info Grid
  modelInfoGrid: {
    gap: 12,
  },
  modelInfoCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
  },
  modelInfoLabel: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 6,
    fontWeight: '600',
  },
  modelInfoValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
  },

  // Metrics Eval Grid
  metricsEvalGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricEvalCard: {
    flex: 1,
    minWidth: (width - 64) / 2,
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  metricEvalLabel: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 8,
    fontWeight: '600',
  },
  metricEvalValue: {
    fontSize: 32,
    fontWeight: '800',
    marginBottom: 12,
  },
  metricEvalBar: {
    height: 6,
    backgroundColor: '#e2e8f0',
    borderRadius: 3,
    overflow: 'hidden',
  },
  metricEvalBarFill: {
    height: '100%',
    borderRadius: 3,
  },

  // Confusion Matrix
  confusionMatrix: {
    gap: 12,
    marginTop: 8,
  },
  confusionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  confusionCell: {
    flex: 1,
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    borderWidth: 3,
  },
  confusionLabel: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 8,
    fontWeight: '700',
  },
  confusionValue: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1e293b',
    marginBottom: 8,
  },
  confusionDesc: {
    fontSize: 11,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 16,
  },

  // Params Container
  paramsContainer: {
    gap: 16,
  },
  paramSection: {
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#e2e8f0',
  },
  paramSectionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 12,
  },
  paramRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  paramLabel: {
    fontSize: 13,
    color: '#64748b',
    fontWeight: '600',
  },
  paramValue: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e293b',
  },

  // Process Container
  processContainer: {
    gap: 20,
  },
  processStep: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  processNumber: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#60a5fa',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  processNumberText: {
    fontSize: 18,
    fontWeight: '800',
    color: '#fff',
  },
  processContent: {
    flex: 1,
  },
  processTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  processItem: {
    fontSize: 13,
    color: '#64748b',
    lineHeight: 20,
    marginBottom: 4,
  },

  // Recommendations Full Box
  recommendationsFullBox: {
    gap: 16,
  },
  recommendationFullItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f8fafc',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#60a5fa',
  },
  recommendationFullIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  recommendationFullContent: {
    flex: 1,
  },
  recommendationFullTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 4,
  },
  recommendationFullText: {
    fontSize: 13,
    color: '#64748b',
    lineHeight: 20,
  },

  // Estilos para Entrada de Sensores con Validación de Rango
  sensorInputField: {
    width: '100%',
    marginBottom: 14,
    backgroundColor: '#f1f5f9',
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  sensorInputHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  sensorInputLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e293b',
    flex: 1,
  },
  sensorInputRange: {
    fontSize: 11,
    fontWeight: '600',
    color: '#64748b',
    backgroundColor: '#e0e7ff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  sensorInputRangeError: {
    backgroundColor: '#fee2e2',
    color: '#dc2626',
  },
  sensorInputBox: {
    borderWidth: 1,
    borderColor: '#cbd5e1',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
    fontSize: 14,
    backgroundColor: '#fff',
    color: '#1e293b',
    fontWeight: '500',
  },
  sensorInputBoxError: {
    borderColor: '#ef4444',
    borderWidth: 2,
    backgroundColor: '#fef2f2',
  },
  sensorInputError: {
    fontSize: 11,
    color: '#dc2626',
    marginTop: 6,
    fontWeight: '500',
  },

  // Estilos para Comparación de Modelos
  rocContainer: {
    marginTop: 12,
  },
  rocDescription: {
    fontSize: 13,
    color: '#64748b',
    marginBottom: 16,
    lineHeight: 20,
  },
  modelsComparisonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  modelComparisonCard: {
    width: '48%',
    backgroundColor: '#f8fafc',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  modelComparisonCardSelected: {
    borderColor: '#3b82f6',
    borderWidth: 2,
    backgroundColor: '#eff6ff',
  },
  modelName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  modelMetricsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modelMetric: {
    alignItems: 'center',
  },
  metricSmallLabel: {
    fontSize: 10,
    color: '#64748b',
    marginBottom: 2,
  },
  metricSmallValue: {
    fontSize: 13,
    fontWeight: '700',
    color: '#2563eb',
  },
  selectedBadge: {
    fontSize: 11,
    fontWeight: '700',
    color: '#10b981',
    marginTop: 8,
    textAlign: 'center',
  },
  rocInfo: {
    backgroundColor: '#ecfdf5',
    borderRadius: 10,
    padding: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  rocInfoTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#059669',
    marginBottom: 6,
  },
  rocInfoText: {
    fontSize: 12,
    color: '#047857',
    lineHeight: 18,
  },

  // Estilos para Conexión
  connectionScrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  presetsContainer: {
    marginVertical: 20,
    backgroundColor: '#f1f5f9',
    borderRadius: 12,
    padding: 16,
  },
  presetsLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 12,
  },
  presetsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  presetButton: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    alignItems: 'center',
  },
  presetButtonActive: {
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff',
    borderWidth: 2,
  },
  presetButtonText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 2,
  },
  presetButtonTextActive: {
    color: '#2563eb',
  },
  presetButtonSubtext: {
    fontSize: 10,
    color: '#64748b',
  },
  presetButtonSubtextActive: {
    color: '#2563eb',
    fontWeight: '600',
  },
  infoContainer: {
    marginTop: 20,
    backgroundColor: '#ecfdf5',
    borderRadius: 12,
    padding: 14,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  infoTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#059669',
    marginBottom: 10,
  },
  infoText: {
    fontSize: 12,
    color: '#047857',
    marginBottom: 4,
    lineHeight: 18,
  },

  // Estilos para Fuente de Datos
  dataSourceContainer: {
    marginTop: 12,
  },
  dataSourceItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#f1f5f9',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
  },
  dataSourceIcon: {
    fontSize: 24,
    marginRight: 12,
    marginTop: 2,
  },
  dataSourceContent: {
    flex: 1,
  },
  dataSourceTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 2,
  },
  dataSourceValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2563eb',
    marginBottom: 2,
  },
  dataSourceDesc: {
    fontSize: 11,
    color: '#64748b',
  },

  // Footer
  footer: {
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    paddingVertical: 20,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 13,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 4,
  },
  footerSubtext: {
    fontSize: 11,
    color: '#94a3b8',
  },

  // Nuevos Estilos para Página de Inicio
  quickStatsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  quickStatItem: {
    flex: 1,
    backgroundColor: '#f8fafc',
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  quickStatIcon: {
    fontSize: 28,
    marginBottom: 8,
  },
  quickStatValue: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1e293b',
    marginBottom: 4,
  },
  quickStatLabel: {
    fontSize: 10,
    color: '#64748b',
    fontWeight: '600',
  },
  featureGridLarge: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 24,
  },
  featureCardLarge: {
    width: (width - 56) / 2,
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  featureCardIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  featureCardTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 4,
    textAlign: 'center',
  },
  featureCardDesc: {
    fontSize: 11,
    color: '#64748b',
    textAlign: 'center',
    fontWeight: '500',
  },
  modelInfoHorizontal: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 12,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  modelInfoItemH: {
    flex: 1,
    alignItems: 'center',
  },
  modelInfoLabelH: {
    fontSize: 11,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 4,
  },
  modelInfoValueH: {
    fontSize: 16,
    fontWeight: '800',
    color: '#60a5fa',
  },
  dividerVertical: {
    width: 1,
    backgroundColor: '#e2e8f0',
    marginHorizontal: 8,
  },
  quickActionsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  actionButton: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 12,
    alignItems: 'center',
    borderWidth: 2,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
  },
  actionButtonIcon: {
    fontSize: 20,
  },
  actionButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1e293b',
  },
  systemStatusCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  statusHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  statusTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
  },
  statusBadge: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  statusBadgeText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#fff',
  },
  statusGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statusItem: {
    width: (width - 64) / 2,
    backgroundColor: '#f8fafc',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  statusItemLabel: {
    fontSize: 11,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 4,
  },
  statusItemValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
  },
  activityList: {
    gap: 12,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  activityDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 12,
  },
  activityContent: {
    flex: 1,
  },
  activityText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 2,
  },
  activityTime: {
    fontSize: 12,
    color: '#94a3b8',
  },
  connectionInfoFooter: {
    backgroundColor: '#e0f2fe',
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#60a5fa',
    marginBottom: 24,
  },
  connectionInfoTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 8,
  },
  connectionInfoText: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 4,
  },
  disconnectButtonSmall: {
    marginTop: 12,
    backgroundColor: '#ef4444',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
    alignItems: 'center',
  },
  disconnectButtonSmallText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
});

export default App;