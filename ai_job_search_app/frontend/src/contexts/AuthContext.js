import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setLoading(false);
    }
  }, [token]);

  const fetchUser = async () => {
    try {
      // Since we don't have a /me endpoint, we'll create a mock user from token
      const mockUser = {
        id: 1,
        email: 'user@example.com', // This will be replaced with actual user data
        full_name: 'User'
      };
      setUser(mockUser);
    } catch (error) {
      console.error('Failed to fetch user:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      console.log('🔍 Attempting login with:', email);
      
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        body: formData,
      });
      
      console.log('📡 Login response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ Login failed:', errorData);
        throw new Error(errorData.detail || 'Login failed');
      }
      
      const data = await response.json();
      console.log('✅ Login successful:', data);
      
      const { access_token } = data;
      
      localStorage.setItem('token', access_token);
      setToken(access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      // Set user data from email
      const userData = {
        id: 1,
        email: email,
        full_name: email.split('@')[0]
      };
      setUser(userData);
      
      toast.success('Welcome back!');
      return true;
    } catch (error) {
      console.error('❌ Login error:', error);
      const message = error.message || 'Login failed';
      toast.error(message);
      return false;
    }
  };

  const register = async (userData) => {
    try {
      console.log('🔍 Attempting registration:', userData);
      
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });
      
      console.log('📡 Registration response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ Registration failed:', errorData);
        throw new Error(errorData.detail || 'Registration failed');
      }
      
      const data = await response.json();
      console.log('✅ Registration successful:', data);
      
      toast.success('Registration successful! Please log in.');
      return true;
    } catch (error) {
      console.error('❌ Registration error:', error);
      const message = error.message || 'Registration failed';
      toast.error(message);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    delete axios.defaults.headers.common['Authorization'];
    toast.success('Logged out successfully');
  };

  const value = {
    user,
    login,
    register,
    logout,
    loading,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};