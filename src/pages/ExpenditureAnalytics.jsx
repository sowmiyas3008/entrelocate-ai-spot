
import { useState, useEffect } from "react";
import { 
  DollarSign, 
  TrendingUp, 
  TrendingDown,
  Calendar,
  Edit3,
  Plus,
  X,
  BarChart3,
  Save,
  ArrowLeft,
  LineChart
} from "lucide-react";
import { auth, db } from "@/firebase/firebase";
import {
  collection,
  doc,
  getDocs,
  getDoc,
  setDoc,
  query
} from "firebase/firestore";

const ExpenditureAnalytics = () => {
  const [ownerName, setOwnerName] = useState("");
  const [userEmail, setUserEmail] = useState("");
  
  // Form Data
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [expenses, setExpenses] = useState('');
  const [outcome, setOutcome] = useState('');
  const [fixedExpenses, setFixedExpenses] = useState(0);
  const [notes, setNotes] = useState(''); // NEW: Add notes
  
  // Results
  const [profitOrLoss, setProfitOrLoss] = useState(null);
  const [result, setResult] = useState('');
  
  // Chart Data
  const [chartData, setChartData] = useState([]);
  
  // Fixed Expenses Modal
  const [popupVisible, setPopupVisible] = useState(false);
  const [newExpenses, setNewExpenses] = useState([{ name: "", amount: "" }]);
  
  // Date Filter
  const [dateFilter, setDateFilter] = useState('7days'); // NEW: Date filter

  // Fetch user data and fixed expenses on load
  useEffect(() => {
    const fetchUserData = async () => {
      const user = auth.currentUser;
      if (!user) {
        console.log("No user logged in");
        setOwnerName("Business Owner");
        return;
      }
      
      setUserEmail(user.email);
      try {
        const userRef = doc(db, "users", user.email);
        const docSnap = await getDoc(userRef);
        if (docSnap.exists()) {
          const data = docSnap.data();
          setOwnerName(data.ownerName || data.shopName || user.email);
        } else {
          setOwnerName(user.email);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
        setOwnerName(user.email);
      }
    };

    fetchUserData();
    fetchFixedExpenses();
    fetchUserExpenditureData();
  }, []);

  // Fetch fixed expenses
  const fetchFixedExpenses = async () => {
    const user = auth.currentUser;
    if (!user) {
      console.log("No user for fixed expenses");
      return;
    }
    
    try {
      const fixedRef = doc(db, "users", user.email, "fixed", "details");
      const fixedSnap = await getDoc(fixedRef);
      
      if (fixedSnap.exists()) {
        const fixedData = fixedSnap.data();
        const expenses = fixedData.expenses || [];
        
        const totalFixed = expenses.reduce((total, expense) => {
          return total + (parseFloat(expense.amount) || 0);
        }, 0);
        
        setFixedExpenses(totalFixed);
        setNewExpenses(expenses.length > 0 ? expenses : [{ name: "", amount: "" }]);
      } else {
        setFixedExpenses(0);
        setNewExpenses([{ name: "", amount: "" }]);
      }
    } catch (error) {
      console.error("Error fetching fixed expenses:", error);
      setFixedExpenses(0);
      setNewExpenses([{ name: "", amount: "" }]);
    }
  };

  // Fetch expenditure history
  const fetchUserExpenditureData = async () => {
    const user = auth.currentUser;
    if (!user) {
      console.log("No user for expenditure data");
      return;
    }
    
    try {
      const userQuery = query(collection(db, `users/${user.email}/expenditure`));
      const querySnapshot = await getDocs(userQuery);
      const data = querySnapshot.docs.map((doc) => ({
        ...doc.data(),
        timestamp: new Date(doc.data().timestamp),
      }));
      
      // Apply date filter
      let daysAgo = 7;
      if (dateFilter === '30days') daysAgo = 30;
      if (dateFilter === '90days') daysAgo = 90;
      if (dateFilter === 'all') daysAgo = 365 * 10; // All time
      
      const filterDate = new Date();
      filterDate.setDate(filterDate.getDate() - daysAgo);
      const filteredData = data.filter((entry) => entry.timestamp >= filterDate);

      setChartData(filteredData.map((entry) => ({
        date: entry.startDate,
        profitLoss: entry.profitOrLoss,
        expenses: entry.expenses + entry.fixedExpenses,
        outcome: entry.outcome,
        notes: entry.notes || '',
      })));
    } catch (error) {
      console.error('Error fetching expenditure data:', error);
      setChartData([]);
    }
  };

  // Re-fetch when date filter changes
  useEffect(() => {
    if (userEmail) {
      fetchUserExpenditureData();
    }
  }, [dateFilter]);

  // Calculate and save
  const calculateAndStore = async () => {
    if (!startDate || !endDate || !expenses || !outcome) {
      alert("Please fill in all fields");
      return;
    }

    const expenseValue = parseFloat(expenses) || 0;
    const outcomeValue = parseFloat(outcome) || 0;
    
    const totalExpenses = fixedExpenses + expenseValue;
    const profitOrLossValue = outcomeValue - totalExpenses;
    const resultValue = profitOrLossValue >= 0 ? "Profit" : "Loss";
    
    const data = {
      startDate,
      endDate,
      expenses: expenseValue,
      fixedExpenses,
      outcome: outcomeValue,
      profitOrLoss: profitOrLossValue,
      result: resultValue,
      timestamp: new Date().toISOString(),
      notes: notes, // Save notes
    };
    
    try {
      const user = auth.currentUser;
      if (user) {
        const docRef = doc(db, `users/${user.email}/expenditure`, new Date().getTime().toString());
        
        await setDoc(docRef, data);
        
        setProfitOrLoss(profitOrLossValue);
        setResult(resultValue);
        
        alert("Data saved successfully!");
        fetchUserExpenditureData();
        
        // Clear form
        setStartDate('');
        setEndDate('');
        setExpenses('');
        setOutcome('');
        setNotes('');
      } else {
        alert("No user is signed in. Please sign in to save data.");
      }
    } catch (error) {
      console.error("Error saving data:", error);
      alert("Error saving data. Please try again.");
    }
  };

  // Calculate summary statistics
  const calculateSummary = () => {
    if (chartData.length === 0) return null;
    
    const totalRevenue = chartData.reduce((sum, entry) => sum + entry.outcome, 0);
    const totalExpenses = chartData.reduce((sum, entry) => sum + entry.expenses, 0);
    const totalProfit = chartData.reduce((sum, entry) => sum + entry.profitLoss, 0);
    const avgDailyProfit = totalProfit / chartData.length;
    
    const bestDay = chartData.reduce((best, entry) => 
      entry.profitLoss > best.profitLoss ? entry : best
    );
    
    const worstDay = chartData.reduce((worst, entry) => 
      entry.profitLoss < worst.profitLoss ? entry : worst
    );
    
    const profitMargin = totalRevenue > 0 ? ((totalProfit / totalRevenue) * 100) : 0;
    
    return {
      totalRevenue,
      totalExpenses,
      totalProfit,
      avgDailyProfit,
      bestDay,
      worstDay,
      profitMargin,
      profitableDays: chartData.filter(d => d.profitLoss > 0).length,
      totalDays: chartData.length
    };
  };

  const summary = calculateSummary();

  // Save fixed expenses
  const handleSaveExpenses = async () => {
    const user = auth.currentUser;
    if (!user) return;

    const validExpenses = newExpenses.filter(expense => 
      expense.name.trim() !== "" && expense.amount.trim() !== ""
    );

    const userRef = doc(db, "users", user.email, "fixed", "details");
    try {
      await setDoc(userRef, { expenses: validExpenses }, { merge: true });
      
      const totalFixed = validExpenses.reduce((total, expense) => {
        return total + (parseFloat(expense.amount) || 0);
      }, 0);
      
      setFixedExpenses(totalFixed);
      alert("Fixed expenses saved successfully!");
      setPopupVisible(false);
    } catch (error) {
      console.error("Error saving expenses:", error);
      alert("Error saving expenses. Please try again.");
    }
  };

  const addNewExpenseRow = () => {
    setNewExpenses([...newExpenses, { name: "", amount: "" }]);
  };

  const removeExpenseRow = (index) => {
    if (newExpenses.length > 1) {
      const updatedExpenses = newExpenses.filter((_, i) => i !== index);
      setNewExpenses(updatedExpenses);
    }
  };

  // Calculate summary metrics
  const totalExpensesAmount = fixedExpenses + (parseFloat(expenses) || 0);
  const maxValue = chartData.length > 0 
    ? Math.max(...chartData.map(d => Math.max(Math.abs(d.profitLoss), d.outcome, d.expenses)))
    : 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-12">
      <div className="container mx-auto max-w-6xl">
        
        {/* Header */}
        <div className="mb-8 flex justify-between items-start">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              Expenditure Analytics
            </h1>
            <p className="text-gray-600">
              Track your business expenses and profitability - {ownerName}
            </p>
          </div>
          
          {/* Date Filter */}
          <div className="bg-white rounded-lg shadow-md p-3">
            <label className="text-sm font-medium text-gray-700 mr-3">View:</label>
            <select
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
              <option value="90days">Last 90 Days</option>
              <option value="all">All Time</option>
            </select>
          </div>
        </div>

        {/* Summary Statistics */}
        {summary && (
          <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl shadow-lg p-6 mb-8 text-white">
            <h2 className="text-2xl font-bold mb-4">📊 Business Overview</h2>
            <div className="grid md:grid-cols-4 gap-6">
              <div>
                <p className="text-blue-100 text-sm mb-1">Total Revenue</p>
                <p className="text-3xl font-bold">₹{summary.totalRevenue.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm mb-1">Total Profit</p>
                <p className="text-3xl font-bold">₹{summary.totalProfit.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm mb-1">Avg Daily Profit</p>
                <p className="text-3xl font-bold">₹{summary.avgDailyProfit.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-blue-100 text-sm mb-1">Profit Margin</p>
                <p className="text-3xl font-bold">{summary.profitMargin.toFixed(1)}%</p>
              </div>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-blue-500">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                <span className="text-sm">Best Day: {summary.bestDay.date} (+₹{summary.bestDay.profitLoss.toFixed(0)})</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingDown className="h-5 w-5" />
                <span className="text-sm">Worst Day: {summary.worstDay.date} (₹{summary.worstDay.profitLoss.toFixed(0)})</span>
              </div>
            </div>
          </div>
        )}

        {/* Metrics Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-600">
            <div className="flex items-center justify-between mb-2">
              <DollarSign className="h-8 w-8 text-blue-600" />
              <button
                onClick={() => setPopupVisible(true)}
                className="text-blue-600 hover:text-blue-700"
              >
                <Edit3 className="h-5 w-5" />
              </button>
            </div>
            <p className="text-sm text-gray-600 mb-1">Fixed Expenses</p>
            <p className="text-3xl font-bold text-gray-900">
              ₹{fixedExpenses.toFixed(2)}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
            <DollarSign className="h-8 w-8 text-orange-500 mb-2" />
            <p className="text-sm text-gray-600 mb-1">Total Expenses</p>
            <p className="text-3xl font-bold text-gray-900">
              ₹{totalExpensesAmount.toFixed(2)}
            </p>
          </div>

          <div className={`bg-white rounded-xl shadow-lg p-6 border-l-4 ${
            profitOrLoss !== null 
              ? profitOrLoss >= 0 
                ? 'border-green-500' 
                : 'border-red-500'
              : 'border-gray-300'
          }`}>
            {profitOrLoss !== null ? (
              profitOrLoss >= 0 ? (
                <TrendingUp className="h-8 w-8 text-green-500 mb-2" />
              ) : (
                <TrendingDown className="h-8 w-8 text-red-500 mb-2" />
              )
            ) : (
              <DollarSign className="h-8 w-8 text-gray-400 mb-2" />
            )}
            <p className="text-sm text-gray-600 mb-1">
              {profitOrLoss !== null 
                ? profitOrLoss >= 0 
                  ? 'Profit' 
                  : 'Loss'
                : 'Profit/Loss'}
            </p>
            <p className={`text-3xl font-bold ${
              profitOrLoss !== null 
                ? profitOrLoss >= 0 
                  ? 'text-green-600' 
                  : 'text-red-600'
                : 'text-gray-900'
            }`}>
              ₹{profitOrLoss !== null ? Math.abs(profitOrLoss).toFixed(2) : '0.00'}
            </p>
          </div>
        </div>

        {/* Input Form */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Add Expenditure Entry
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Calendar className="inline h-4 w-4 mr-1" />
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Calendar className="inline h-4 w-4 mr-1" />
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Variable Expenses (₹)
              </label>
              <input
                type="number"
                value={expenses}
                onChange={(e) => setExpenses(e.target.value)}
                placeholder="Enter variable expenses"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Outcome Gained (₹)
              </label>
              <input
                type="number"
                value={outcome}
                onChange={(e) => setOutcome(e.target.value)}
                placeholder="Enter income/revenue"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notes (Optional)
              </label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g., Holiday sale, Equipment repair"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <button
            onClick={calculateAndStore}
            className="mt-6 w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center justify-center gap-2"
          >
            <Save className="h-5 w-5" />
            Calculate and Save
          </button>
        </div>

        {/* Visual Chart */}
        {chartData.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <LineChart className="h-6 w-6 text-blue-600" />
              <h3 className="text-2xl font-bold text-gray-900">
                Profit/Loss Trend
              </h3>
            </div>

            <div className="relative h-64 flex items-end justify-around gap-4 border-b-2 border-gray-300 pb-2">
              {chartData.map((entry, index) => {
                const absValue = Math.abs(entry.profitLoss);
                const heightPercentage = maxValue > 0 ? (absValue / maxValue) * 80 : 10;
                const isProfit = entry.profitLoss >= 0;
                
                return (
                  <div key={index} className="flex-1 flex flex-col items-center max-w-[100px]">
                    <div className="relative w-full flex flex-col items-center justify-end h-full">
                      {/* Amount label on top */}
                      <div className={`absolute -top-10 left-1/2 transform -translate-x-1/2 text-xs font-bold whitespace-nowrap ${
                        isProfit ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {isProfit ? '+' : '-'}₹{absValue.toFixed(0)}
                      </div>
                      
                      {/* Bar */}
                      <div
                        className={`w-full rounded-t-lg transition-all cursor-pointer hover:opacity-80 ${
                          isProfit ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{ 
                          height: `${Math.max(heightPercentage, 10)}%`,
                          minHeight: '20px'
                        }}
                        title={`${isProfit ? 'Profit' : 'Loss'}: ₹${absValue.toFixed(2)}`}
                      />
                    </div>
                    
                    {/* Date label below */}
                    <div className="text-xs text-gray-600 mt-3 text-center font-medium">
                      {entry.date ? new Date(entry.date).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' }) : 'N/A'}
                    </div>
                  </div>
                );
              })}
            </div>
            
            <div className="flex justify-center gap-6 mt-8">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm text-gray-600 font-medium">Profit</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm text-gray-600 font-medium">Loss</span>
              </div>
            </div>
          </div>
        )}

        {/* Chart Section */}
        {chartData.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="h-6 w-6 text-blue-600" />
              <h3 className="text-2xl font-bold text-gray-900">
                Recent Analytics
              </h3>
            </div>

            <div className="space-y-4">
              {chartData.map((entry, index) => (
                <div key={index} className="border-l-4 border-blue-600 pl-4 py-3 bg-gray-50 rounded-r-lg">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="font-semibold text-gray-900">{entry.date}</p>
                        {entry.notes && (
                          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                            📝 {entry.notes}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600">
                        Revenue: ₹{entry.outcome.toFixed(2)} | Expenses: ₹{entry.expenses.toFixed(2)}
                      </p>
                    </div>
                    <div className={`text-xl font-bold ${
                      entry.profitLoss >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {entry.profitLoss >= 0 ? '+' : ''}₹{entry.profitLoss.toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="mt-8">
          <button
            onClick={() => window.history.back()}
            className="bg-gray-100 text-gray-700 py-3 px-6 rounded-lg hover:bg-gray-200 transition-colors font-medium flex items-center gap-2"
          >
            <ArrowLeft className="h-5 w-5" />
            Back
          </button>
        </div>
      </div>

      {/* Fixed Expenses Modal */}
      {popupVisible && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Manage Fixed Expenses
              </h2>
              <button
                onClick={() => setPopupVisible(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="space-y-4 mb-6">
              {newExpenses.map((expense, index) => (
                <div key={index} className="flex gap-3 items-center">
                  <input
                    type="text"
                    placeholder="Expense Name (e.g., Rent)"
                    value={expense.name}
                    onChange={(e) => {
                      const updatedExpenses = [...newExpenses];
                      updatedExpenses[index].name = e.target.value;
                      setNewExpenses(updatedExpenses);
                    }}
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <input
                    type="number"
                    placeholder="Amount (₹)"
                    value={expense.amount}
                    onChange={(e) => {
                      const updatedExpenses = [...newExpenses];
                      updatedExpenses[index].amount = e.target.value;
                      setNewExpenses(updatedExpenses);
                    }}
                    className="w-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {newExpenses.length > 1 && (
                    <button
                      onClick={() => removeExpenseRow(index)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="flex gap-3">
              <button
                onClick={addNewExpenseRow}
                className="flex-1 bg-gray-100 text-gray-700 py-3 px-6 rounded-lg hover:bg-gray-200 transition-colors font-medium flex items-center justify-center gap-2"
              >
                <Plus className="h-5 w-5" />
                Add Row
              </button>
              <button
                onClick={handleSaveExpenses}
                className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center justify-center gap-2"
              >
                <Save className="h-5 w-5" />
                Save Expenses
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExpenditureAnalytics;