import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, MapPin, Building2, ExternalLink, Clock, DollarSign, Briefcase } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const JobSearch = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [location, setLocation] = useState('');
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchType, setSearchType] = useState('manual'); // 'manual' or 'auto'

  const handleManualSearch = async (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) {
      setError('Please enter a search term');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const params = new URLSearchParams({
        keyword: searchTerm,
        ...(location && { location })
      });

      const response = await fetch(`http://localhost:8000/api/jobs/search?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to search jobs');
      }

      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (err) {
      setError('Failed to search jobs. Please try again.');
      console.error('Job search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAutomatedSearch = async () => {
    setLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const params = new URLSearchParams({
        ...(location && { location }),
        max_jobs: '20'
      });

      const response = await fetch(`http://localhost:8000/api/jobs/auto-search?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to perform automated search');
      }

      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (err) {
      setError('Failed to perform automated search. Please upload your CV first.');
      console.error('Automated search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const truncateText = (text, maxLength = 200) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold gradient-text mb-2">Job Search</h1>
          <p className="text-gray-400 text-lg">
            Find your next opportunity with AI-powered job matching
          </p>
        </motion.div>

        {/* Search Type Toggle */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-6 mb-6"
        >
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <button
              onClick={() => setSearchType('manual')}
              className={`flex-1 py-3 px-4 rounded-lg transition-all ${
                searchType === 'manual'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Search className="inline-block mr-2" size={20} />
              Manual Search
            </button>
            <button
              onClick={() => setSearchType('auto')}
              className={`flex-1 py-3 px-4 rounded-lg transition-all ${
                searchType === 'auto'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Briefcase className="inline-block mr-2" size={20} />
              AI-Powered Search
            </button>
          </div>

          {searchType === 'manual' ? (
            <form onSubmit={handleManualSearch} className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Job Title or Keywords
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                    <input
                      type="text"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      placeholder="e.g. Software Engineer, Data Scientist"
                      className="w-full pl-10 pr-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Location (Optional)
                  </label>
                  <div className="relative">
                    <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="e.g. San Francisco, Remote"
                      className="w-full pl-10 pr-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full sm:w-auto px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? <LoadingSpinner size="sm" /> : 'Search Jobs'}
              </button>
            </form>
          ) : (
            <div className="text-center py-6">
              <h3 className="text-lg font-semibold text-white mb-2">
                AI-Powered Job Search
              </h3>
              <p className="text-gray-400 mb-6">
                Let our AI find jobs that match your CV skills and experience automatically
              </p>
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Preferred Location (Optional)
                  </label>
                  <div className="relative">
                    <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      placeholder="e.g. San Francisco, Remote"
                      className="w-full pl-10 pr-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
              <button
                onClick={handleAutomatedSearch}
                disabled={loading}
                className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? <LoadingSpinner size="sm" /> : 'Start AI Search'}
              </button>
            </div>
          )}
        </motion.div>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mb-6"
          >
            <p className="text-red-400">{error}</p>
          </motion.div>
        )}

        {/* Results */}
        {jobs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">
                Found {jobs.length} job{jobs.length !== 1 ? 's' : ''}
              </h2>
            </div>

            <div className="space-y-4">
              {jobs.map((job, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                  className="card p-6 hover:border-blue-500/30 transition-all duration-200"
                >
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-2">
                        {job.title}
                      </h3>
                      <div className="flex items-center text-gray-400 mb-2">
                        <Building2 size={16} className="mr-2" />
                        <span>{job.company}</span>
                        {job.location && (
                          <>
                            <MapPin size={16} className="ml-4 mr-2" />
                            <span>{job.location}</span>
                          </>
                        )}
                      </div>
                      <div className="flex items-center text-sm text-gray-500">
                        <span className="bg-blue-900/30 text-blue-400 px-2 py-1 rounded-full text-xs">
                          {job.source}
                        </span>
                      </div>
                    </div>
                  </div>

                  {job.description && (
                    <div className="mb-4">
                      <p className="text-gray-300 leading-relaxed">
                        {truncateText(job.description.replace(/<[^>]*>/g, ''))}
                      </p>
                    </div>
                  )}

                  <div className="flex justify-between items-center">
                    <div className="flex items-center text-sm text-gray-400">
                      <Clock size={16} className="mr-2" />
                      <span>Recently posted</span>
                    </div>
                    <button className="btn-secondary">
                      <ExternalLink size={16} className="mr-2" />
                      View Details
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {!loading && jobs.length === 0 && !error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <Search className="mx-auto h-16 w-16 text-gray-600 mb-4" />
            <h3 className="text-xl font-semibold text-gray-400 mb-2">
              No jobs found
            </h3>
            <p className="text-gray-500">
              {searchType === 'auto' 
                ? 'Try uploading your CV first or adjust your location preferences'
                : 'Try different keywords or location'
              }
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default JobSearch;