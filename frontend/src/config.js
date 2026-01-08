/**
 * Configuracao da aplicacao
 * Em producao, defina VITE_API_URL no arquivo .env.production
 */

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
