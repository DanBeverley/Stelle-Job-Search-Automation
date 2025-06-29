import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, CheckCircle, XCircle, Loader } from 'lucide-react';

const APITest = () => {
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState({});

  const API_BASE = 'http://localhost:8000';

  const tests = [
    {
      id: 'health',
      name: 'Health Check',
      endpoint: '/api/health',
      method: 'GET',
      description: 'Test if backend is running'
    },
    {
      id: 'salary',
      name: 'Salary Prediction',
      endpoint: '/api/salary/predict',
      method: 'POST',
      body: {
        job_title: 'Software Engineer',
        experience_level: 'Mid',
        location: 'San Francisco'
      },
      description: 'Test AI salary prediction'
    },
    {
      id: 'cover_letter',
      name: 'Cover Letter Generation',
      endpoint: '/api/cover-letter/generate',
      method: 'POST',
      body: {
        job_title: 'Software Engineer',
        company: 'Google',
        additional_info: '5 years Python experience'
      },
      description: 'Test AI cover letter generation'
    },
    {
      id: 'interview',
      name: 'Interview Response',
      endpoint: '/api/interview/generate-response',
      method: 'POST',
      body: {
        question: 'Tell me about yourself'
      },
      description: 'Test AI interview preparation'
    }
  ];

  const runTest = async (test) => {
    setLoading(prev => ({ ...prev, [test.id]: true }));
    
    try {
      const options = {
        method: test.method,
        headers: {
          'Content-Type': 'application/json',
        },
      };

      if (test.body) {
        options.body = JSON.stringify(test.body);
      }

      const response = await fetch(`${API_BASE}${test.endpoint}`, options);
      const data = await response.json();

      setResults(prev => ({
        ...prev,
        [test.id]: {
          success: response.ok,
          status: response.status,
          data: data,
          timestamp: new Date().toLocaleTimeString()
        }
      }));
    } catch (error) {
      setResults(prev => ({
        ...prev,
        [test.id]: {
          success: false,
          error: error.message,
          timestamp: new Date().toLocaleTimeString()
        }
      }));
    } finally {
      setLoading(prev => ({ ...prev, [test.id]: false }));
    }
  };

  const runAllTests = async () => {
    for (const test of tests) {
      await runTest(test);
      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  };

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold gradient-text mb-2">
            Backend API Testing
          </h1>
          <p className="text-gray-400 text-lg">
            Test the AI Job Search backend functionality
          </p>
        </motion.div>

        <div className="mb-6">
          <button
            onClick={runAllTests}
            className="btn-primary flex items-center space-x-2"
            disabled={Object.values(loading).some(Boolean)}
          >
            <Play size={20} />
            <span>Run All Tests</span>
          </button>
        </div>

        <div className="space-y-6">
          {tests.map((test, index) => (
            <motion.div
              key={test.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-xl font-semibold text-white">{test.name}</h3>
                  <p className="text-gray-400 text-sm">{test.description}</p>
                  <p className="text-gray-500 text-xs font-mono">
                    {test.method} {test.endpoint}
                  </p>
                </div>
                
                <div className="flex items-center space-x-3">
                  {results[test.id] && (
                    <div className="flex items-center space-x-2">
                      {results[test.id].success ? (
                        <CheckCircle className="text-green-400" size={20} />
                      ) : (
                        <XCircle className="text-red-400" size={20} />
                      )}
                      <span className="text-xs text-gray-400">
                        {results[test.id].timestamp}
                      </span>
                    </div>
                  )}
                  
                  <button
                    onClick={() => runTest(test)}
                    disabled={loading[test.id]}
                    className="btn-secondary flex items-center space-x-2"
                  >
                    {loading[test.id] ? (
                      <Loader className="animate-spin" size={16} />
                    ) : (
                      <Play size={16} />
                    )}
                    <span>Test</span>
                  </button>
                </div>
              </div>

              {test.body && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Request Body:</h4>
                  <pre className="bg-dark-800 text-gray-300 p-3 rounded text-xs overflow-x-auto">
                    {JSON.stringify(test.body, null, 2)}
                  </pre>
                </div>
              )}

              {results[test.id] && (
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2">
                    Response {results[test.id].success ? 
                      `(${results[test.id].status})` : 
                      '(Error)'
                    }:
                  </h4>
                  <div className="bg-dark-800 text-gray-300 p-3 rounded text-xs overflow-x-auto max-h-48 overflow-y-auto">
                    {results[test.id].success ? (
                      <pre>{JSON.stringify(results[test.id].data, null, 2)}</pre>
                    ) : (
                      <div className="text-red-400">
                        Error: {results[test.id].error}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        <div className="mt-8 p-6 bg-dark-800/30 rounded-lg border border-dark-600">
          <h3 className="text-lg font-semibold text-white mb-2">Backend Status</h3>
          <p className="text-gray-400 text-sm mb-4">
            Make sure your backend is running on http://localhost:8000
          </p>
          <div className="text-xs text-gray-500">
            <p>Start backend: <code className="bg-dark-700 px-2 py-1 rounded">cd ai_job_search_app\backend && python -m uvicorn main:app --reload --port 8000</code></p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APITest;