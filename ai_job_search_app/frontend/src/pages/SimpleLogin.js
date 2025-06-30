import React, { useState } from 'react';

const SimpleLogin = () => {
  const [email, setEmail] = useState('test@example.com');
  const [password, setPassword] = useState('test123');
  const [message, setMessage] = useState('');

  const testRegister = async () => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ Registration successful: ${data.email}`);
      } else {
        const error = await response.json();
        setMessage(`❌ Registration failed: ${error.detail}`);
      }
    } catch (error) {
      setMessage(`❌ Registration error: ${error.message}`);
    }
  };

  const testLogin = async () => {
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ Login successful! Token: ${data.access_token.substring(0, 50)}...`);
        localStorage.setItem('token', data.access_token);
      } else {
        const error = await response.json();
        setMessage(`❌ Login failed: ${error.detail}`);
      }
    } catch (error) {
      setMessage(`❌ Login error: ${error.message}`);
    }
  };

  const testForgotPassword = async () => {
    try {
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ Forgot password: ${data.message}`);
      } else {
        const error = await response.json();
        setMessage(`❌ Forgot password failed: ${error.detail}`);
      }
    } catch (error) {
      setMessage(`❌ Forgot password error: ${error.message}`);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Simple Auth Test</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <div style={{ marginBottom: '10px' }}>
          <label>Email: </label>
          <input 
            type="email" 
            value={email} 
            onChange={(e) => setEmail(e.target.value)}
            style={{ padding: '5px', width: '200px' }}
          />
        </div>
        <div style={{ marginBottom: '10px' }}>
          <label>Password: </label>
          <input 
            type="password" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)}
            style={{ padding: '5px', width: '200px' }}
          />
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <button onClick={testRegister} style={{ marginRight: '10px', padding: '10px' }}>
          Test Register
        </button>
        <button onClick={testLogin} style={{ marginRight: '10px', padding: '10px' }}>
          Test Login
        </button>
        <button onClick={testForgotPassword} style={{ padding: '10px' }}>
          Test Forgot Password
        </button>
      </div>

      <div style={{ 
        padding: '10px', 
        backgroundColor: '#f0f0f0', 
        border: '1px solid #ccc',
        minHeight: '100px',
        whiteSpace: 'pre-wrap'
      }}>
        {message || 'Click buttons to test...'}
      </div>
    </div>
  );
};

export default SimpleLogin;