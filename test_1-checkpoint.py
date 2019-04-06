{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pandas import read_csv, DataFrame\n",
    "import statsmodels.api as sm\n",
    "import statsmodels as statsmodels\n",
    "from statsmodels.iolib.table import SimpleTable\n",
    "from sklearn.metrics import r2_score\n",
    "import ml_metrics as metrics\n",
    "import matplotlib as plt\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import random\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>col</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>11000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>12500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>16900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>13500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>15000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>16000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>17000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>18500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>19900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>11</td>\n",
       "      <td>23500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>12</td>\n",
       "      <td>35000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>13</td>\n",
       "      <td>40000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>14</td>\n",
       "      <td>51000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>15</td>\n",
       "      <td>62500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>16</td>\n",
       "      <td>76900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>17</td>\n",
       "      <td>83500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18</td>\n",
       "      <td>95000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    date    col\n",
       "0      1  10000\n",
       "1      2  11000\n",
       "2      3  12500\n",
       "3      4  16900\n",
       "4      5  13500\n",
       "5      6  15000\n",
       "6      7  16000\n",
       "7      8  17000\n",
       "8      9  18500\n",
       "9     10  19900\n",
       "10    11  23500\n",
       "11    12  35000\n",
       "12    13  40000\n",
       "13    14  51000\n",
       "14    15  62500\n",
       "15    16  76900\n",
       "16    17  83500\n",
       "17    18  95000"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataset = pd.read_csv('2.csv')\n",
    "display(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>col</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>11000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>12500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>16900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>13500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>15000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>16000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>17000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>18500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>19900</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   date    col\n",
       "0     1  10000\n",
       "1     2  11000\n",
       "2     3  12500\n",
       "3     4  16900\n",
       "4     5  13500\n",
       "5     6  15000\n",
       "6     7  16000\n",
       "7     8  17000\n",
       "8     9  18500\n",
       "9    10  19900"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Creating train and test set \n",
    "#Index 10392 marks the end of October 2013 \n",
    "#train=dataset[['Дата','Длит']][0:8000]\n",
    "#train['Дата'] = pd.to_datetime(train['Дата'])\n",
    "#train['Длит'] = pd.to_numeric(train['Длит'])\n",
    "#train['Длит'] = pd.to_datetime(train['Длит'], format='H%:M%')\n",
    "#test=dataset[['Дата','Длит']][8001:9000]\n",
    "#test['Дата'] = pd.to_datetime(test['Дата'])\n",
    "#train['Длит'] = pd.to_numeric(train['Длит'])\n",
    "#test['Длит'] = pd.to_datetime(test['Длит'], format='H%:M%') \n",
    "train = dataset[:10]\n",
    "test = dataset[10:]\n",
    "display(train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Метод Holt\n",
    "## https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# train = train.resample('D').mean() \n",
    "# test = test.resample('D').mean()\n",
    "\n",
    "# #Plotting data\n",
    "# train.Count.plot(figsize=(15,8))\n",
    "# test.Count.plot(figsize=(15,8))\n",
    "# plt.show()\n",
    "import matplotlib.pyplot as plt \n",
    "import statsmodels.api as sm\n",
    "from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt\n",
    "\n",
    "y_hat_avg = test.copy()\n",
    "\n",
    "fit1 = Holt(np.asarray(train['col'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)\n",
    "\n",
    "y_hat_avg['Holt_linear'] = fit1.forecast(len(test))\n",
    "\n",
    "plt.figure(figsize=(16,8))\n",
    "plt.plot(train['col'], label='Train')\n",
    "plt.plot(test['col'], label='Test')\n",
    "plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')\n",
    "plt.legend(loc='best')\n",
    "plt.show()\n",
    "#print(y_hat_avg.keys)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Holt-Winters Method\n",
    "## https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    date    col   Holt_Winter\n",
      "10    11  23500  20752.306350\n",
      "11    12  35000  21742.284026\n",
      "12    13  40000  22732.261702\n",
      "13    14  51000  23722.239378\n",
      "14    15  62500  24712.217054\n",
      "15    16  76900  25702.194730\n",
      "16    17  83500  26692.172406\n",
      "17    18  95000  27682.150082\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA7UAAAHVCAYAAAAuMtxGAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xl4VNXh//H3SQgk7JDEBdCKQhXcUKIorq0Laq2iKKigqFSrrUsXW7Xt91tt+21ra2vdurigVhBUEMUVt1btD7eguOKCG+IaAghIgCz398cMKZgAARJOZub9eh4eZu7cO/lMHsOTj+fcc0KSJEiSJEmSlInyYgeQJEmSJGlDWWolSZIkSRnLUitJkiRJyliWWkmSJElSxrLUSpIkSZIylqVWkiRJkpSxLLWSJEmSpIxlqZUkSZIkZSxLrSRJkiQpY7WJHWBDlZSUJNtss03sGJIkSZKkZjZjxox5SZKUNuXcjC2122yzDeXl5bFjSJIkSZKaWQjhg6ae6/RjSZIkSVLGstRKkiRJkjKWpVaSJEmSlLEy9p7axlRXVzN37lyWLVsWO0rOKCwspFevXhQUFMSOIkmSJCkHZVWpnTt3Lp06dWKbbbYhhBA7TtZLkoTKykrmzp1L7969Y8eRJEmSlIOyavrxsmXLKC4uttBuIiEEiouLHRmXJEmSFE1WlVrAQruJ+f2WJEmSFFPWlVpJkiRJUu6w1DaTyspKBgwYwIABA9hiiy3o2bNn/fMVK1Y06T1OO+003nzzzRZOKkmSJEnZI6sWioqpuLiYmTNnAnDJJZfQsWNHLrjggtXOSZKEJEnIy2v8/yXcdNNNLZ5TkiRJkrJJ1pbaS+99jdc/XtSs79m/R2d++e0d1+ua2bNnM3ToUPbdd1+effZZ7rvvPi699FJeeOEFqqqqGDFiBP/7v/8LwL777ss111zDTjvtRElJCWeddRYPPvgg7du355577mGzzTZr1s8jSZIkSZnO6cebwOuvv86YMWN48cUX6dmzJ7///e8pLy/npZde4pFHHuH1119vcM0XX3zBAQccwEsvvcTee+/N2LFjIySXJEmSpNYta0dq13dEtSVtt9127LHHHvXPJ0yYwI033khNTQ0ff/wxr7/+Ov3791/tmqKiIg4//HAABg4cyFNPPbVJM0uSJElSJsjaUtuadOjQof7x22+/zZVXXslzzz1H165dGTVqVKP7vLZt27b+cX5+PjU1NZskqyRJkiRlEqcfb2KLFi2iU6dOdO7cmU8++YRp06bFjiRJkiRJGcuR2k1s9913p3///uy0005su+227LPPPrEjSZIkSVLGCkmSxM6wQcrKypLy8vLVjs2aNYt+/fpFSpS7/L5LkiRJGWT+e9B1a8jLj51kjUIIM5IkKWvKuU4/liRJkqRcMf9duPEQeOji2EmajaVWkiRJknLBkgq49Vioq4U9z4idptl4T60kSZIkZbvlS+C24bD4Uxh9L5T0jZ2o2VhqJUmSJCmb1VbDnafCJzNhxHjYao/YiZqVpVaSJEmSslWSwH0/gNmPwJF/gR2OiJ2o2XlPrSRJkiRlq3/9Fl4cBwdcCGWnxU7TIiy1zaSyspIBAwYwYMAAtthiC3r27Fn/fMWKFU1+n7Fjx/Lpp5+2YFJJkiRJOaF8LDz5B9jtZDgwe1Y7/iqnHzeT4uJiZs6cCcAll1xCx44dueCCC9b7fcaOHcvuu+/OFlts0dwRJUmSJOWKN+6H+38MfYekph2HEDtRi8neUvvgRfDpK837nlvsDIf/fr0vu+WWW7j22mtZsWIFgwcP5pprrqGuro7TTjuNmTNnkiQJZ555JptvvjkzZ85kxIgRFBUV8dxzz9G2bdvm/QySJEmSstucZ2HS6dBjNzj+JsjP3toH2VxqW4lXX32VKVOmMH36dNq0acOZZ57JxIkT2W677Zg3bx6vvJIq3gsXLqRr165cffXVXHPNNQwYMCByckmSJEkZp+ItmDACOveEk+6Ath1iJ2px2VtqN2BEtSU8+uijPP/885SVlQFQVVXFVlttxZAhQ3jzzTc5//zzOeKIIzj00EMjJ5UkSZKU0RZ9AuOOhbw2MGoydCiJnWiTyN5S20okScLpp5/Or3/96wavvfzyyzz44INcddVVTJ48meuuuy5CQkmSJEkZb9kXMP54qFoAp94H3XvHTrTJuPpxCzv44IO54447mDdvHpBaJXnOnDlUVFSQJAnHH388l156KS+88AIAnTp1YvHixTEjS5IkScokNSvg9lFQMQuG/zN1L20OcaS2he2888788pe/5OCDD6auro6CggL+/ve/k5+fz5gxY0iShBACl112GQCnnXYa3/nOd1woSpIkSdK61dXB3WfDe0/C0L9Dn4NiJ9rkQpIksTNskLKysqS8vHy1Y7NmzaJfv36REuUuv++SJElSJA//AqZfDQf9Evb7Uew0zSaEMCNJkrKmnOv0Y0mSJEnKRE//NVVo9zgD9v1h7DTRWGolSZIkKdO8OhmmXQz9vg2HXwYhxE4UjaVWkiRJkjLJe0/ClLNg673h2OshLz92oqgstZIkSZKUKT57DSaOhO7bwokToKAodqLoLLWSJEmSlAkWfgjjhkHbjjByEhR1i52oVXBLH0mSJElq7ZbOTxXaFV/C6Q9B161iJ2o1LLWSJEmS1JpVV8HEk2DBezDqLth8x9iJWhWnHzezjh07rvb85ptv5pxzzlnrNZdccgmXX355/fkff/zxGs+95557GDp0aP3z3/3ud/Tp06f++b333stRRx0FwBFHHMHChQvX+rXX9fUkSZIkRVRXC3edAXOehmP+Ab33i52o1cnakdrLnruMN+a/0azvuUP3Hbhwzwub9T2/6uabb2annXaiR48ejb4+ePBgzjzzzPrnTz/9NJ07d+bzzz9ns802Y/r06eyzzz4APPDAAxv99RpTU1NDmzZZ+5+OJEmS1DokCTx4Icy6F4b8DnY6NnaiVsmR2k3ogw8+4KCDDmKXXXbhoIMOYs6cOau9PmnSJMrLyxk5ciQDBgygqqqqwXuUlpbSpUsXZs+eDcBHH33EsGHDmD59OgDTp09n8ODBAGyzzTbMmzeP999/n379+nHGGWew4447cuihh1JVVdXo15sxYwYHHHAAAwcOZMiQIXzyyScAHHjggfzsZz/jgAMO4Morr2zJb5MkSZIkgP9cAc9fD4PPhb2/FztNq5W1w20tPaK6JlVVVQwYMKD++fz58+unA59zzjmccsopjB49mrFjx3Leeedx991315973HHHcc0113D55ZdTVla2xq8xePBgpk+fTm1tLX379mWvvfZi2rRpHHnkkbz88svsscceDa55++23mTBhAtdffz3Dhw9n8uTJjBo1arWvV11dzbnnnss999xDaWkpt99+Oz//+c8ZO3YsAAsXLuSJJ55orm+VJEmSpDWZOQEeuxR2Ph4O/lXsNK1a1pbaWIqKipg5c2b985tvvpny8nIgNVX4rrvuAuDkk0/mpz/96QZ9jX322ae+1O69997sueee/OpXv+LFF19k++23p7CwsME1vXv3ri/bAwcO5P33329w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\n",
      "text/plain": [
       "<Figure size 1152x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_hat_avg = test.copy()\n",
    "fit1 = ExponentialSmoothing(np.asarray(train['col']) ,trend='add', ).fit()\n",
    "y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))\n",
    "print(y_hat_avg[])\n",
    "plt.figure(figsize=(16,8))\n",
    "plt.plot( train['col'], label='Train')\n",
    "plt.plot(test['col'], label='Test')\n",
    "plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')\n",
    "plt.legend(loc='best')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ARIMA\n",
    "## https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Non-stationary starting autoregressive parameters found with `enforce_stationarity` set to True.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-91-05575170f86f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0my_hat_avg\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtest\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mfit1\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msm\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtsa\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstatespace\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSARIMAX\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrain\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'col'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;31m# y_hat_avg['SARIMA'] = fit1.predict(start=\"2013-11-1\", end=\"2013-12-31\", dynamic=True)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;31m# plt.figure(figsize=(16,8))\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;31m# plt.plot( train['col'], label='Train')\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\semen\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\statsmodels\\tsa\\statespace\\mlemodel.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, start_params, transformed, cov_type, cov_kwds, method, maxiter, full_output, disp, callback, return_params, optim_score, optim_complex_step, optim_hessian, flags, **kwargs)\u001b[0m\n\u001b[0;32m    430\u001b[0m         \"\"\"\n\u001b[0;32m    431\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mstart_params\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 432\u001b[1;33m             \u001b[0mstart_params\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart_params\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    433\u001b[0m             \u001b[0mtransformed\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    434\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\semen\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\statsmodels\\tsa\\statespace\\sarimax.py\u001b[0m in \u001b[0;36mstart_params\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    997\u001b[0m         )\n\u001b[0;32m    998\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0minvalid_ar\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 999\u001b[1;33m             raise ValueError('Non-stationary starting autoregressive'\n\u001b[0m\u001b[0;32m   1000\u001b[0m                              \u001b[1;34m' parameters found with `enforce_stationarity`'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1001\u001b[0m                              ' set to True.')\n",
      "\u001b[1;31mValueError\u001b[0m: Non-stationary starting autoregressive parameters found with `enforce_stationarity` set to True."
     ]
    }
   ],
   "source": [
    "y_hat_avg = test.copy()\n",
    "fit1 = sm.tsa.statespace.SARIMAX(train['col']).fit()\n",
    "# y_hat_avg['SARIMA'] = fit1.predict(start=\"2013-11-1\", end=\"2013-12-31\", dynamic=True)\n",
    "# plt.figure(figsize=(16,8))\n",
    "# plt.plot( train['col'], label='Train')\n",
    "# plt.plot(test['col'], label='Test')\n",
    "# plt.plot(y_hat_avg['SARIMA'], label='SARIMA')\n",
    "# plt.legend(loc='best')\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
