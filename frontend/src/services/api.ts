import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Satisfaction Prediction
  predictSatisfactionV1: async (data: any) => {
    const response = await axiosInstance.post('/satisfaction/v1/predict', data);
    return response.data;
  },

  // Purchase Prediction (New function added)
  predictPurchaseV1: async (data: any) => {
    const response = await axiosInstance.post('/purchase/v1/predict', data);
    return response.data;
  },
  
  predictPurchaseV2: async (data: any) => {
    const response = await axiosInstance.post('/purchase/v2/predict', data);
    return response.data;
  },
  
  predictSatisfactionV2: async (data: any) => {
    const response = await axiosInstance.post('/satisfaction/final/predict', data);
    return response.data;
  },

  // Price Recommendation
  getPriceRecommendation: async (data: any) => {
    const response = await axiosInstance.post('/price/recommend', data);
    return response.data;
  },

  // Product Recommendation
  getProductRecommendations: async (customerId: string) => {
    const response = await axiosInstance.get(`/recommend/products/${customerId}`);
    return response.data;
  },

  // Business Insights
  getBusinessMetrics: async () => {
    const response = await axiosInstance.get('/metrics/business');
    return response.data;
  },

  // Customer Segmentation
  getCustomerSegments: async () => {
    const response = await axiosInstance.get('/segments/customers');
    return response.data;
  },

  // Error handler wrapper
  errorHandler: (error: any) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.data);
      throw new Error(error.response.data.message || 'An error occurred');
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.request);
      throw new Error('Network error - please check your connection');
    } else {
      // Other errors
      console.error('Error:', error.message);
      throw error;
    }
  }
};