/// Navigation operation mode
enum NavMode {
  /// Pure inertial dead reckoning during GNSS outage
  deadReckoning,

  /// GNSS-aided fusion mode with active satellite corrections
  gnssAided,

  /// Initializing / calibrating phone mount orientation
  calibrating,
}
