import React, { useState } from 'react';

const AuthTest = () => {
  const [results, setResults] = useState([]);
  const [email, setEmail] = useState('test@example.com');
  const [password, setPassword] = useState('testpassword');

  const addResult = (test, status, message) => {
    setResults(prev => [...prev, { test, status, message, time: new Date().toLocaleTimeString() }]);
  };

  const testRegistration = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, password: password }),
      });
      
      const data = await response.json();
      if (response.ok) {
        addResult('Registration', 'success', 'User registered successfully');
      } else {
        addResult('Registration', 'error', data.detail || 'Registration failed');
      }
    } catch (error) {
      addResult('Registration', 'error', error.message);
    }
  };

  const testLogin = async () => {
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (response.ok) {
        addResult('Login', 'success', `Token received: ${data.access_token.substring(0, 20)}...`);
        localStorage.setItem('token', data.access_token);
      } else {
        addResult('Login', 'error', data.detail || 'Login failed');
      }
    } catch (error) {
      addResult('Login', 'error', error.message);
    }
  };

  const testForgotPassword = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email }),
      });
      
      const data = await response.json();
      if (response.ok) {
        addResult('Forgot Password', 'success', data.message);
      } else {
        addResult('Forgot Password', 'error', data.detail || 'Failed');
      }
    } catch (error) {
      addResult('Forgot Password', 'error', error.message);
    }
  };

  const testJobSearch = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/jobs/test-auto-search', {
        method: 'GET',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json' 
        },
      });
      
      const data = await response.json();
      if (response.ok) {
        addResult('Job Search', 'success', `Found ${data.jobs?.length || 0} jobs`);
      } else {
        addResult('Job Search', 'error', data.detail || 'Failed');
      }
    } catch (error) {
      addResult('Job Search', 'error', error.message);
    }
  };

  const clearResults = () => setResults([]);

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-8">Authentication & API Test</h1>
        
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Test Credentials</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              className="p-3 bg-gray-700 text-white rounded border border-gray-600"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              className="p-3 bg-gray-700 text-white rounded border border-gray-600"
            />
          </div>
          
          <div className="flex flex-wrap gap-4">
            <button onClick={testRegistration} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
              Test Registration
            </button>
            <button onClick={testLogin} className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
              Test Login
            </button>
            <button onClick={testForgotPassword} className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700">
              Test Forgot Password
            </button>
            <button onClick={testJobSearch} className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700">
              Test Job Search
            </button>
            <button onClick={clearResults} className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
              Clear Results
            </button>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Test Results</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {results.map((result, index) => (
              <div
                key={index}
                className={`p-3 rounded border-l-4 ${
                  result.status === 'success' 
                    ? 'bg-green-900/20 border-green-500 text-green-300' 
                    : 'bg-red-900/20 border-red-500 text-red-300'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <span className="font-semibold">{result.test}:</span> {result.message}
                  </div>
                  <span className="text-xs opacity-70">{result.time}</span>
                </div>
              </div>
            ))}
            {results.length === 0 && (
              <p className="text-gray-400 text-center py-8">No tests run yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthTest;