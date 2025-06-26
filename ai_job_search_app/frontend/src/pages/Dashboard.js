import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Upload,
  Search,
  FileText,
  DollarSign,
  TrendingUp,
  PenTool,
  MessageSquare,
  BarChart3,
  Calendar,
  Clock,
  CheckCircle,
  AlertCircle,
  Plus,
  ArrowRight
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    applications: 0,
    interviews: 0,
    cvUploaded: false,
    lastActivity: null
  });
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Mock data for now - replace with actual API calls
      setTimeout(() => {
        setStats({
          applications: 12,
          interviews: 3,
          cvUploaded: true,
          lastActivity: new Date().toISOString()
        });
        
        setRecentActivity([
          {
            id: 1,
            type: 'application',
            title: 'Applied to Software Engineer at TechCorp',
            time: '2 hours ago',
            status: 'pending'
          },
          {
            id: 2,
            type: 'interview',
            title: 'Interview scheduled with StartupXYZ',
            time: '1 day ago',
            status: 'scheduled'
          },
          {
            id: 3,
            type: 'cover_letter',
            title: 'Generated cover letter for Data Scientist role',
            time: '2 days ago',
            status: 'completed'
          }
        ]);
        
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setLoading(false);
    }
  };

  const quickActions = [
    {
      title: 'Upload CV',
      description: 'Parse and analyze your resume',
      icon: Upload,
      link: '/cv-upload',
      color: 'from-blue-500 to-cyan-500',
      disabled: false
    },
    {
      title: 'Search Jobs',
      description: 'Find relevant opportunities',
      icon: Search,
      link: '/job-search',
      color: 'from-purple-500 to-pink-500',
      disabled: !stats.cvUploaded
    },
    {
      title: 'Predict Salary',
      description: 'AI-powered salary estimation',
      icon: DollarSign,
      link: '/salary-prediction',
      color: 'from-green-500 to-emerald-500',
      disabled: !stats.cvUploaded
    },
    {
      title: 'Analyze Skills',
      description: 'Identify skill gaps',
      icon: TrendingUp,
      link: '/skill-analysis',
      color: 'from-orange-500 to-red-500',
      disabled: !stats.cvUploaded
    },
    {
      title: 'Cover Letter',
      description: 'Generate personalized letters',
      icon: PenTool,
      link: '/cover-letter',
      color: 'from-indigo-500 to-purple-500',
      disabled: !stats.cvUploaded
    },
    {
      title: 'Interview Prep',
      description: 'Practice with AI questions',
      icon: MessageSquare,
      link: '/interview-prep',
      color: 'from-teal-500 to-blue-500',
      disabled: !stats.cvUploaded
    }
  ];

  const getActivityIcon = (type) => {
    switch (type) {
      case 'application':
        return FileText;
      case 'interview':
        return Calendar;
      case 'cover_letter':
        return PenTool;
      default:
        return CheckCircle;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'pending':
        return 'text-yellow-400';
      case 'scheduled':
        return 'text-blue-400';
      default:
        return 'text-gray-400';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="xl" />
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold gradient-text mb-2">
            Welcome back, {user?.full_name || user?.email}!
          </h1>
          <p className="text-gray-400 text-lg">
            Your AI-powered job search command center
          </p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Applications</p>
                <p className="text-3xl font-bold text-white">{stats.applications}</p>
              </div>
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <FileText className="text-white" size={24} />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Interviews</p>
                <p className="text-3xl font-bold text-white">{stats.interviews}</p>
              </div>
              <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                <Calendar className="text-white" size={24} />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">CV Status</p>
                <p className="text-lg font-semibold text-white">
                  {stats.cvUploaded ? 'Uploaded' : 'Not Uploaded'}
                </p>
              </div>
              <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                stats.cvUploaded 
                  ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                  : 'bg-gradient-to-r from-orange-500 to-red-500'
              }`}>
                {stats.cvUploaded ? (
                  <CheckCircle className="text-white" size={24} />
                ) : (
                  <AlertCircle className="text-white" size={24} />
                )}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Success Rate</p>
                <p className="text-3xl font-bold text-white">25%</p>
              </div>
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="text-white" size={24} />
              </div>
            </div>
          </motion.div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Quick Actions */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="card p-6 mb-8"
            >
              <h2 className="text-2xl font-bold text-white mb-6">Quick Actions</h2>
              <div className="grid md:grid-cols-2 gap-4">
                {quickActions.map((action, index) => {
                  const Icon = action.icon;
                  return (
                    <motion.div
                      key={action.title}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + index * 0.1 }}
                    >
                      {action.disabled ? (
                        <div className="p-4 rounded-lg bg-dark-800/50 border border-dark-600 opacity-50 cursor-not-allowed">
                          <div className="flex items-center mb-3">
                            <div className={`w-10 h-10 bg-gradient-to-r ${action.color} rounded-lg flex items-center justify-center mr-3`}>
                              <Icon size={20} className="text-white" />
                            </div>
                            <div>
                              <h3 className="font-semibold text-white">{action.title}</h3>
                              <p className="text-sm text-gray-400">{action.description}</p>
                            </div>
                          </div>
                          {!stats.cvUploaded && (
                            <p className="text-xs text-orange-400">Upload CV first</p>
                          )}
                        </div>
                      ) : (
                        <Link
                          to={action.link}
                          className="block p-4 rounded-lg bg-dark-800/30 border border-dark-600 hover:border-primary-500/30 hover:bg-dark-800/60 transition-all duration-200 group"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center">
                              <div className={`w-10 h-10 bg-gradient-to-r ${action.color} rounded-lg flex items-center justify-center mr-3`}>
                                <Icon size={20} className="text-white" />
                              </div>
                              <div>
                                <h3 className="font-semibold text-white group-hover:text-primary-300 transition-colors">
                                  {action.title}
                                </h3>
                                <p className="text-sm text-gray-400">{action.description}</p>
                              </div>
                            </div>
                            <ArrowRight 
                              size={16} 
                              className="text-gray-400 group-hover:text-primary-400 group-hover:translate-x-1 transition-all" 
                            />
                          </div>
                        </Link>
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          </div>

          {/* Recent Activity */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">Recent Activity</h2>
                <Link to="/applications" className="text-primary-400 hover:text-primary-300 text-sm font-medium">
                  View All
                </Link>
              </div>

              <div className="space-y-4">
                {recentActivity.map((activity, index) => {
                  const Icon = getActivityIcon(activity.type);
                  return (
                    <motion.div
                      key={activity.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.8 + index * 0.1 }}
                      className="flex items-start space-x-3 p-3 rounded-lg bg-dark-800/30 hover:bg-dark-800/50 transition-colors"
                    >
                      <div className="w-8 h-8 bg-dark-700 rounded-lg flex items-center justify-center flex-shrink-0">
                        <Icon size={16} className="text-gray-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white truncate">
                          {activity.title}
                        </p>
                        <div className="flex items-center space-x-2 mt-1">
                          <p className="text-xs text-gray-400">{activity.time}</p>
                          <span className={`text-xs ${getStatusColor(activity.status)} capitalize`}>
                            {activity.status}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}

                {recentActivity.length === 0 && (
                  <div className="text-center py-8">
                    <Clock className="mx-auto h-12 w-12 text-gray-600 mb-4" />
                    <p className="text-gray-400">No recent activity</p>
                    <p className="text-sm text-gray-500 mt-1">Start by uploading your CV</p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;