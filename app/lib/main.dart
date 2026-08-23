import 'package:flutter/material.dart';
import 'ui/debug_dashboard.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const IdrNavApp());
}

class IdrNavApp extends StatelessWidget {
  const IdrNavApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IDR-Nav Engine',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF38BDF8),
          secondary: Color(0xFF34D399),
          surface: Color(0xFF1E293B),
        ),
      ),
      home: const DebugDashboard(),
    );
  }
}
