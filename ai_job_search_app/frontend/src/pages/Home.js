import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Upload,
  Search,
  DollarSign,
  TrendingUp,
  PenTool,
  MessageSquare,
  Shield,
  Zap,
  Target,
  ArrowRight,
  CheckCircle,
  Star
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const Home = () => {
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: Upload,
      title: 'Smart CV Processing',
      description: 'AI-powered parsing of PDFs, DOCX, and images with OCR technology',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Search,
      title: 'Multi-Provider Job Search',
      description: 'Aggregate jobs from multiple sources with intelligent matching',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: DollarSign,
      title: 'Salary Prediction',
      description: 'XGBoost model trained on 30K+ records with high accuracy',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: TrendingUp,
      title: 'Skill Gap Analysis',
      description: 'Identify missing skills and get personalized recommendations',
      color: 'from-orange-500 to-red-500'
    },
    {
      icon: PenTool,
      title: 'AI Cover Letters',
      description: 'Fine-tuned language model for personalized cover letters',
      color: 'from-indigo-500 to-purple-500'
    },
    {
      icon: MessageSquare,
      title: 'Interview Preparation',
      description: 'AI-powered question generation and STAR method analysis',
      color: 'from-teal-500 to-blue-500'
    }
  ];

  const stats = [
    { value: '30K+', label: 'Training Records', icon: Target },
    { value: '95%', label: 'Accuracy Rate', icon: CheckCircle },
    { value: '59+', label: 'Test Coverage', icon: Shield },
    { value: '4.9', label: 'User Rating', icon: Star }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-dark-950 via-navy-900 to-dark-900 pt-20 pb-32">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23334155" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-50"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-5xl md:text-7xl font-bold mb-6"
            >
              <span className="gradient-text">AI-Powered</span><br />
              Job Search Automation
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
            >
              Revolutionize your job search with intelligent CV processing, salary predictions, 
              and AI-generated cover letters. Built for the modern job seeker.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              {isAuthenticated ? (
                <Link to="/dashboard" className="btn-primary btn-lg group">
                  Go to Dashboard
                  <ArrowRight className="ml-2 group-hover:translate-x-1 transition-transform" size={20} />
                </Link>
              ) : (
                <>
                  <Link to="/register" className="btn-primary btn-lg group">
                    Get Started Free
                    <ArrowRight className="ml-2 group-hover:translate-x-1 transition-transform" size={20} />
                  </Link>
                  <Link to="/login" className="btn-secondary btn-lg">
                    Sign In
                  </Link>
                </>
              )}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-dark-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="text-center"
                >
                  <div className="flex justify-center mb-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-navy-600 rounded-lg flex items-center justify-center">
                      <Icon size={24} className="text-white" />
                    </div>
                  </div>
                  <div className="text-3xl font-bold gradient-text mb-2">{stat.value}</div>
                  <div className="text-gray-400">{stat.label}</div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-dark-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="section-title">Powerful Features</h2>
            <p className="section-subtitle">
              Everything you need to supercharge your job search with AI
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="card-hover p-6"
                >
                  <div className="flex items-center mb-4">
                    <div className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-lg flex items-center justify-center mr-4`}>
                      <Icon size={24} className="text-white" />
                    </div>
                    <h3 className="text-xl font-semibold">{feature.title}</h3>
                  </div>
                  <p className="text-gray-400">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary-600 to-navy-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.h2
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-white mb-6"
          >
            Ready to Transform Your Job Search?
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-blue-100 mb-8"
          >
            Join thousands of job seekers who have already upgraded their career journey with AI
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            {!isAuthenticated && (
              <Link to="/register" className="btn bg-white text-primary-600 hover:bg-gray-100 btn-lg font-semibold">
                Start Your Free Trial
              </Link>
            )}
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;