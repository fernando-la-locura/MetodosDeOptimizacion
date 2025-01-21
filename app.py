import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { AlertCircle, Calculator, Info } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function EquationSolver() {
  const [equations, setEquations] = useState('2x1 + 3x2 = 18\nx1 - x2 = 1');
  const [solution, setSolution] = useState(null);
  const [error, setError] = useState('');
  const [currentMethod, setCurrentMethod] = useState('cramer');

  const validateEquations = (eqs) => {
    const lines = eqs.trim().split('\n').filter(line => line.trim());
    if (lines.length < 2) return "Ingrese al menos 2 ecuaciones";
    if (!lines.every(line => line.includes('='))) return "Cada línea debe contener una ecuación con '='";
    return "";
  };

  const handleSolve = () => {
    const validationError = validateEquations(equations);
    if (validationError) {
      setError(validationError);
      setSolution(null);
      return;
    }
    
    // Simulamos la solución para el ejemplo
    if (currentMethod === 'cramer') {
      setSolution({
        method: 'Método de Cramer',
        variables: {
          x1: 4.2,
          x2: 3.2
        },
        steps: [
          'Determinante de A: -5',
          'Determinante de A1: -21',
          'Determinante de A2: -16'
        ]
      });
    }
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-6 w-6" />
            Resolutor de Sistemas de Ecuaciones Lineales
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="input" className="space-y-4">
            <TabsList>
              <TabsTrigger value="input">Entrada</TabsTrigger>
              <TabsTrigger value="help">Ayuda</TabsTrigger>
            </TabsList>

            <TabsContent value="input" className="space-y-4">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Ingrese sus ecuaciones (una por línea):
                  </label>
                  <textarea 
                    className="w-full min-h-[200px] p-3 border rounded-md font-mono"
                    value={equations}
                    onChange={(e) => setEquations(e.target.value)}
                    placeholder="Ejemplo:&#13;&#10;2x1 + 3x2 = 18&#13;&#10;x1 - x2 = 1"
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex gap-4">
                    <select 
                      className="p-2 border rounded-md"
                      value={currentMethod}
                      onChange={(e) => setCurrentMethod(e.target.value)}
                    >
                      <option value="cramer">Método de Cramer</option>
                      <option value="gauss">Gauss-Jordan</option>
                      <option value="substitution">Sustitución</option>
                    </select>
                    <Button onClick={handleSolve}>Resolver</Button>
                  </div>

                  {error && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="help">
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  <h3 className="font-semibold mb-2">Instrucciones:</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Escriba cada ecuación en una línea separada</li>
                    <li>Use el formato: ax1 + bx2 + cx3 = d</li>
                    <li>Las variables deben escribirse como x1, x2, x3, etc.</li>
                    <li>Ejemplo: 2x1 + 3x2 = 18</li>
                  </ul>
                </AlertDescription>
              </Alert>
            </TabsContent>
          </Tabs>

          {solution && (
            <div className="mt-6 space-y-4">
              <h3 className="text-lg font-semibold">{solution.method}</h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(solution.variables).map(([variable, value]) => (
                  <Card key={variable}>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{value.toFixed(4)}</div>
                      <div className="text-sm text-gray-500">{variable}</div>
                    </CardContent>
                  </Card>
                ))}
              </div>
              <div className="mt-4">
                <h4 className="font-medium mb-2">Pasos:</h4>
                <ul className="space-y-2">
                  {solution.steps.map((step, index) => (
                    <li key={index} className="bg-gray-50 p-2 rounded">{step}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}