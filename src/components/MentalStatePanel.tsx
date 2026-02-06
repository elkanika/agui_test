import React, { useEffect, useState } from 'react';
import { Brain, FileText, Activity, Save, RefreshCw } from 'lucide-react';

interface MentalStatePanelProps {
  studentInfo: {
    curso: string;
    universidad: string;
    nivel: string;
  };
  onUpdateStudentInfo: (info: any) => void;
  isOpen: boolean;
}

export default function MentalStatePanel({ studentInfo, onUpdateStudentInfo, isOpen }: MentalStatePanelProps) {
  const [agentState, setAgentState] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [localInfo, setLocalInfo] = useState(studentInfo);

  // Sync local state when props change
  useEffect(() => {
    setLocalInfo(studentInfo);
  }, [studentInfo]);

  const fetchAgentState = async () => {
    setLoading(true);
    try {
      // Assuming backend is at localhost:8000 based on README
      // In production, this should use an env var or proxy
      const response = await fetch('http://localhost:8000/api/state');
      if (response.ok) {
        const data = await response.json();
        setAgentState(data);
      }
    } catch (e) {
      console.error("Failed to fetch agent state", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchAgentState();
      // Poll every 5 seconds while open
      const interval = setInterval(fetchAgentState, 5000);
      return () => clearInterval(interval);
    }
  }, [isOpen]);

  const handleInfoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setLocalInfo(prev => ({ ...prev, [name]: value }));
  };

  const handleSaveInfo = () => {
    onUpdateStudentInfo(localInfo);
  };

  if (!isOpen) return null;

  return (
    <div className="w-80 bg-gray-50 border-l border-gray-200 h-full overflow-y-auto flex flex-col shadow-xl transition-all duration-300">
      <div className="p-4 bg-white border-b border-gray-200 flex justify-between items-center sticky top-0 z-10">
        <h2 className="font-bold text-gray-800 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-600" />
          Estado Mental
        </h2>
        <button
          onClick={fetchAgentState}
          className={`p-1 rounded-full hover:bg-gray-100 ${loading ? 'animate-spin' : ''}`}
          title="Actualizar"
        >
          <RefreshCw className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="p-4 space-y-6">
        {/* Contexto Activo (Editable) */}
        <section>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1">
            <Activity className="w-3 h-3" />
            Contexto Activo
          </h3>
          <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm space-y-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Curso</label>
              <input
                type="text"
                name="curso"
                value={localInfo.curso}
                onChange={handleInfoChange}
                className="w-full text-sm border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 p-1.5 bg-gray-50"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Universidad</label>
              <input
                type="text"
                name="universidad"
                value={localInfo.universidad}
                onChange={handleInfoChange}
                className="w-full text-sm border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 p-1.5 bg-gray-50"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Nivel</label>
              <input
                type="text"
                name="nivel"
                value={localInfo.nivel}
                onChange={handleInfoChange}
                className="w-full text-sm border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 p-1.5 bg-gray-50"
              />
            </div>
            <button
              onClick={handleSaveInfo}
              className="w-full mt-2 flex items-center justify-center gap-2 bg-purple-600 text-white text-xs font-medium py-1.5 rounded hover:bg-purple-700 transition-colors"
            >
              <Save className="w-3 h-3" />
              Actualizar Contexto
            </button>
          </div>
        </section>

        {/* Pensamiento Reciente (Read-only) */}
        <section>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1">
            <Brain className="w-3 h-3" />
            Último Pensamiento
          </h3>
          <div className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm text-sm space-y-2">
            {agentState?.last_thought && Object.keys(agentState.last_thought).length > 0 ? (
              <>
                <div className="border-b border-gray-100 pb-2">
                  <span className="text-gray-500 text-xs block">Clasificación:</span>
                  <p className="font-medium text-gray-800 whitespace-pre-wrap text-xs mt-1 bg-gray-50 p-1 rounded">
                    {agentState.last_thought.clasificacion || "N/A"}
                  </p>
                </div>
                <div className="border-b border-gray-100 pb-2">
                  <span className="text-gray-500 text-xs block">Búsqueda Generada:</span>
                  <p className="font-medium text-gray-800 text-xs mt-1">
                    "{agentState.last_thought.consulta_busqueda || "N/A"}"
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs block">Docs Encontrados:</span>
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 mt-1">
                    {agentState.last_thought.documentos_encontrados ?? 0}
                  </span>
                </div>
              </>
            ) : (
              <p className="text-gray-400 italic text-xs">El agente aún no ha procesado ninguna consulta.</p>
            )}
          </div>
        </section>

        {/* Documentos en Memoria */}
        <section>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1">
            <FileText className="w-3 h-3" />
            Documentos Cargados
          </h3>
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm divide-y divide-gray-100">
            {agentState?.loaded_pdfs && agentState.loaded_pdfs.length > 0 ? (
              <ul className="max-h-40 overflow-y-auto">
                {agentState.loaded_pdfs.map((pdf: string, idx: number) => (
                  <li key={idx} className="px-3 py-2 text-xs text-gray-600 flex items-center gap-2 hover:bg-gray-50">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                    <span className="truncate" title={pdf}>{pdf}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="p-3 text-center text-gray-400 text-xs italic">
                No hay documentos cargados en memoria activa.
              </div>
            )}

            {agentState?.temario_summary && (
               <div className="p-2 bg-gray-50 text-xs text-gray-500 border-t border-gray-100">
                  <span className="font-semibold">Temario:</span> {agentState.temario_summary}
               </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
