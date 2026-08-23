import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/main.dart';

void main() {
  testWidgets('App renders DebugDashboard smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const IdrNavApp());
    expect(find.text('IDR-Nav Engine Monitor'), findsOneWidget);
  });
}
